# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

import os
import socket

import hydra
import ray
from omegaconf import OmegaConf

from typing import Any, Optional, Union

from datasets import Dataset, IterableDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BaseImageProcessor,
    DataCollator,
    FeatureExtractionMixin,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
)
from trl.data_utils import maybe_apply_chat_template, maybe_extract_prompt

from verl.trainer.dpo.ray_trainer import RayDPOTrainer
#from verl.trainer.dpo.reward import load_reward_manager


@hydra.main(config_path="config", config_name="dpo_trainer", version_base=None)
def main(config):
    run_dpo(config)


# Define a function to run the DPO-like training process
def run_dpo(config) -> None:
    # Check if Ray is not initialized
    if not ray.is_initialized():
        # Initialize Ray with a local cluster configuration
        # Set environment variables in the runtime environment to control tokenizer parallelism,
        # NCCL debug level, VLLM logging level, and allow runtime LoRA updating
        # `num_cpus` specifies the number of CPU cores Ray can use, obtained from the configuration
        TENSORBOARD_DIR = os.environ.get("TENSORBOARD_DIR", "runs")
        ray.init(
            runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", "VLLM_LOGGING_LEVEL": "WARN", "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true", "NVTE_FLASH_ATTN": "1", "TENSORBOARD_DIR": TENSORBOARD_DIR}},
            num_cpus=config.ray_init.num_cpus,
        )

    # Create a remote instance of the TaskRunner class, and
    # Execute the `run` method of the TaskRunner instance remotely and wait for it to complete
    if OmegaConf.select(config.trainer, "profile_steps") is not None and len(OmegaConf.select(config.trainer, "profile_steps")) > 0:
        nsight_options = OmegaConf.to_container(config.trainer.controller_nsight_options)
        runner = TaskRunner.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))

    # [Optional] get the path of the timeline trace file from the configuration, default to None
    # This file is used for performance analysis
    timeline_json_file = config.ray_init.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:
    def run(self, config):
        # Print the initial configuration. `resolve=True` will evaluate symbolic values.
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")

        pprint(OmegaConf.to_container(config, resolve=True))

        OmegaConf.resolve(config)

        # Download the checkpoint from HDFS to the local machine.
        # `use_shm` determines whether to use shared memory, which could lead to faster model loading if turned on
        local_path = copy_to_local(config.actor_ref.model.path, use_shm=config.actor_ref.model.get("use_shm", False)) #按照model路径的模型copy

        # Instantiate the tokenizer and processor.
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code) #加载tokenizer模型
        # Used for multimodal LLM, could be None
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)


        assert config.actor_ref.actor.strategy == "fsdp", f"only fsdp is verified, {config.actor_ref.actor.strategy} may lead to error result."
        # Define worker classes based on the actor strategy.
        if config.actor_ref.actor.strategy in ["fsdp", "fsdp2"]: #目前使用的fsdp
            #assert config.critic.strategy in ["fsdp", "fsdp2"]
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers_dpo import ActorRefWorker #使用的是ActorRefWorker，没有带rollout

            actor_cls = ActorRefWorker
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_ref.actor.strategy == "megatron":
            #assert config.actor_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRefWorker #这个地方还没有修改成针对DPO的，暂时不支持

            actor_cls = ActorRefWorker
            ray_worker_group_cls = NVMegatronRayWorkerGroup

        else:
            raise NotImplementedError

        from verl.trainer.dpo.ray_trainer import ResourcePoolManager, Role

        # Map roles to their corresponding remote worker classes.
        role_worker_mapping = {
            Role.Actor: ray.remote(actor_cls),
        }

        # Define the resource pool specification.
        # Map roles to the resource pool.
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.Actor: global_pool_id,
            #Role.Critic: global_pool_id,
        }


        # Add a reference policy worker if KL loss or KL reward is used.
        #if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
        # DPO中一定会使用ref model
        role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRefWorker)
        mapping[Role.RefPolicy] = global_pool_id

        # Load the reward manager for training and validation.
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        from verl.utils.dataset.dpo_dataset import init_collator, collate_fn #这里的collate_fn集成了原来的逻辑，又加上verl本身的逻辑
        init_collator(tokenizer.pad_token_id) #使用tokenizer中的pad id来初始化会被collate_fn调用到的内容

        # Create training and validation datasets.
        train_dataset, val_dataset = create_rl_dataset(config.data, tokenizer, config, processor)
        train_sampler = create_rl_sampler(config.data, train_dataset)

        # Initialize the PPO trainer.
        trainer = RayDPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn, #按照tensor 和 non-tensor进行了一个切分 和 汇合；
            train_sampler=train_sampler,
            device_name=config.trainer.device,
        )
        # Initialize the workers of the trainer.
        trainer.init_workers()
        # Start the training process.
        trainer.fit()


def create_rl_dataset(data_config, tokenizer, config, processor):
    """Create a dataset.

    Arguments:
        data_paths: List of paths to data files.
        data_config: The data config.
        tokenizer (Tokenizer): The tokenizer.
        processor (Processor): The processor.

    Returns:
        dataset (Dataset): The dataset.
    """
    from torch.utils.data import Dataset

    from verl.utils.dataset.rl_dataset import RLHFDataset
    from verl.utils.dataset.dpo_dataset import create_dpo_dataset

    '''
    # Check if a custom dataset class is specified in the data configuration
    # and if the path to the custom class is provided
    if "custom_cls" in data_config and data_config.custom_cls.get("path", None) is not None:
        from verl.utils.import_utils import load_extern_type

        # Dynamically load the custom dataset class
        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
        # Verify that the custom dataset class inherits from torch.utils.data.Dataset
        if not issubclass(dataset_cls, Dataset):
            raise TypeError(f"The custom dataset class '{data_config.custom_cls.name}' from '{data_config.custom_cls.path}' must inherit from torch.utils.data.Dataset")
    else:
        # Use the default RLHFDataset class if no custom class is specified
        dataset_cls = RLHFDataset
    print(f"Using dataset class: {dataset_cls.__name__}")

    # Instantiate the dataset using the determined dataset class
    dataset = dataset_cls(
        data_files=data_paths,
        tokenizer=tokenizer,
        processor=processor,
        config=data_config,
    )
    '''
    
    dataset = create_dpo_dataset(training_args=None, sft_config=data_config, tokenizer=tokenizer) #training_args本来是要通过main_process_first来加速的，但是现在还没有用，设置为None

    #这里需要用_prepare_dataset来进行dataset的一次加工，这个想比较好地调用不太容易；
    
    train_dataset = dataset['train']
    validation_dataset = dataset['validation']
    
    # processing_class默认用的是tokenizer
    train_dataset = _prepare_dataset(train_dataset, processing_class=tokenizer, args=config.data, dataset_name="train")
    
    validation_dataset = _prepare_dataset(validation_dataset, processing_class=tokenizer, args=config.data, dataset_name="validation")
    
    return train_dataset, validation_dataset

def tokenize_row(
        features: dict[str, str],
        processing_class: PreTrainedTokenizerBase,
        max_prompt_length: Optional[int] = None,
        max_completion_length: Optional[int] = None,
        add_special_tokens: bool = True,
    ) -> dict[str, list[int]]:
    """
    Tokenize a row of the dataset.

    Args:
        features (`dict[str, str]`):
            Row of the dataset, should contain the keys `"prompt"`, `"chosen"`, and `"rejected"`.
        processing_class (`PreTrainedTokenizerBase`):
            Processing class used to process the data.
        max_prompt_length (`int` or `None`):
            Maximum length of the prompt sequence. If `None`, the prompt sequence is not truncated.
        max_completion_length (`int` or `None`):
            Maximum length of the completion sequences. If `None`, the completion sequences are not truncated.
        add_special_tokens (`bool`):
            Whether to add special tokens to the sequences. Typically used for encoder-decoder models. If `True`,
            the prompt sequence will have a bos token prepended and an eos token appended. In any case, the
            completion sequences will have an eos token appended.

    Returns:
        `dict[str, list[int]]`:
            Tokenized sequences with the keys `"prompt_input_ids"`, `"chosen_input_ids"`, and
            `"rejected_input_ids".

    Example:
    ```python
    >>> from transformers import GPT2Tokenizer

    >>> tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    >>> features = {"prompt": "The sky is", "chosen": " blue", "rejected": " green"}
    >>> DPOTrainer.tokenize_row(
    ...     features, tokenizer, max_prompt_length=3, max_completion_length=3, add_special_tokens=False
    ... )
    {'prompt_input_ids': [464, 6766, 318], 'chosen_input_ids': [4171, 50256], 'rejected_input_ids': [4077, 50256]}
    ```
    """
    tokenizer = processing_class  # the processing class is a tokenizer
    prompt_input_ids = tokenizer(features["prompt"], add_special_tokens=False)["input_ids"]
    chosen_input_ids = tokenizer(features["chosen"], add_special_tokens=False)["input_ids"]
    rejected_input_ids = tokenizer(features["rejected"], add_special_tokens=False)["input_ids"]

    # Add special tokens (typically for encoder-decoder models)
    if add_special_tokens:
        if tokenizer.bos_token_id is not None:
            prompt_input_ids = [tokenizer.bos_token_id] + prompt_input_ids
        if tokenizer.eos_token_id is not None:
            prompt_input_ids = prompt_input_ids + [tokenizer.eos_token_id]
    chosen_input_ids = chosen_input_ids + [tokenizer.eos_token_id]
    rejected_input_ids = rejected_input_ids + [tokenizer.eos_token_id]

    # Truncate prompt and completion sequences
    if max_prompt_length is not None:
        prompt_input_ids = prompt_input_ids[-max_prompt_length:] #左截断
    if max_completion_length is not None:
        chosen_input_ids = chosen_input_ids[:max_completion_length] #右截断
        rejected_input_ids = rejected_input_ids[:max_completion_length]

    return {
        "prompt_input_ids": prompt_input_ids,
        "chosen_input_ids": chosen_input_ids,
        "rejected_input_ids": rejected_input_ids,
    }


def _prepare_dataset(
    dataset: Union[Dataset, IterableDataset],
    processing_class: Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin],
    args: Any,
    dataset_name: str,
) -> Union[Dataset, IterableDataset]:
    # Build the kwargs for the `map` function
    map_kwargs = {}
    if isinstance(dataset, Dataset):  # IterableDataset does not support num_proc nor writer_batch_size
        map_kwargs["num_proc"] = args.preprocess_num_workers
        map_kwargs["writer_batch_size"] = 10

    # Extract prompt if needed
    if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
        map_kwargs["desc"] = f"Extracting prompt in {dataset_name} dataset"
    dataset = dataset.map(maybe_extract_prompt, **map_kwargs)

    # Apply the chat template if needed
    if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
        map_kwargs["desc"] = f"Applying chat template to {dataset_name} dataset"
    dataset = dataset.map(
        maybe_apply_chat_template, fn_kwargs={"tokenizer": processing_class, "tools": args.tools}, **map_kwargs
    )

    # Tokenize the dataset
    if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
        map_kwargs["desc"] = f"Tokenizing {dataset_name} dataset"

    dataset = dataset.map(
        tokenize_row,
        remove_columns=["chosen", "rejected"],
        fn_kwargs={
            "processing_class": processing_class,
            "max_prompt_length": args.max_prompt_length, #这里会限制promt内容的长度
            "max_completion_length": args.max_completion_length,
            # for enc-dec, we add the special tokens ([bos_token] + prompt + [eos_token]; completion + [eos_token])
            "add_special_tokens": False,
        },
        **map_kwargs,
    )

    return dataset

def create_rl_sampler(data_config, dataset):
    """Create a sampler for the dataset.

    Arguments:
        data_config: The data config.
        dataset (Dataset): The dataset.

    Returns:
        sampler (Sampler): The sampler.
    """
    import torch
    from torch.utils.data import RandomSampler, SequentialSampler

    # Use a sampler to facilitate checkpoint resumption.
    # If shuffling is enabled in the data configuration, create a random sampler.
    if data_config.shuffle:
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(data_config.get("seed", 1))
        sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)
    else:
        # If shuffling is disabled, use a sequential sampler to iterate through the dataset in order.
        sampler = SequentialSampler(data_source=dataset)

    return sampler


if __name__ == "__main__":
    main()
