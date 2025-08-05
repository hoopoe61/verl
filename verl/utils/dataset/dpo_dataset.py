import os

import datasets
import torch
import transformers
import pdb
import copy
import random
from datasets import Dataset, load_dataset
from dataclasses import dataclass
# from typing import Dict, Optional
from torch.nn.utils.rnn import pad_sequence
from typing import Any, Dict, List, Optional, Tuple, Union
from transformers import DataCollatorForLanguageModeling, PreTrainedModel, PreTrainedTokenizerBase, TrainerCallback
from tqdm import tqdm
import random
from collections import defaultdict
import numpy as np

from trl.trainer.dpo_trainer import DataCollatorForPreference

global collator

def init_collator(padding_value):
    global collator
    collator = DataCollatorForPreference(padding_value)

def collate_fn(data_list: list[dict]) -> dict:
    """
    Collate a batch of sample dicts into batched tensors and arrays.

    Args:
        data_list: List of dicts mapping feature names to torch.Tensor or other values.

    Returns:
        Dict where tensor entries are stacked into a torch.Tensor of shape
        (batch_size, *dims) and non-tensor entries are converted to
        np.ndarray of dtype object with shape (batch_size,).
    """
    global collator
    data_list = collator.torch_call(data_list)
    
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    #for data in data_list:
    for key, val in data_list.items():
        if isinstance(val, torch.Tensor):
            tensors[key].append(val)
        else:
            non_tensors[key].append(val)

    for key, val in tensors.items():
        #tensors[key] = torch.stack(val, dim=0)
        assert len(val)==1, "now only val's len == 1 are supported."
        tensors[key] = val[0]

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    return {**tensors, **non_tensors}


def create_dataset(training_args, sft_config, tokenizer):
    # pdb.set_trace()
    if sft_config.train_file_path:
        train_file_path = sft_config.train_file_path
    elif sft_config.dataset_name:
        # train_file_path = sft_config.dataset_name + "/train.jsonl"
        train_file_path = sft_config.dataset_name 
    if sft_config.validate_file_path:
        validate_file_path = sft_config.validate_file_path
    elif sft_config.dataset_name:
        validate_file_path = '/code/chenzongchao/SFT/train-code/model_save/eval.jsonl'
    raw_datasets = datasets.load_dataset("json", data_files={'train': train_file_path,
                                                             'validation': validate_file_path})

def create_distill_dataset(training_args, sft_config, tokenizer):

    train_file_path = sft_config.train_file_path
    validate_file_path = sft_config.validate_file_path

    raw_datasets = datasets.load_dataset("json", data_files={'train': train_file_path,
                                                             'validation': validate_file_path})
    print(f"Raw train dataset size: {len(raw_datasets['train'])}")
    print(f"Raw validation dataset size: {len(raw_datasets['validation'])}")
    default_sys_prompt = '''<|im_start|>system 你是一个名为"南北阁"的人工智能助手，正在与人类用户进行交谈。你的目标是以最有帮助和最逻辑的方式回答问题，同时确保内容的安全性。你的回答中不应包含任何有害、政治化、宗教化、不道德、种族主义、非法的内容。请确保你的回答不带有社会偏见，符合社会主义价值观。如果遇到的问题无意义或事实上不连贯，请不要回答错误的内容，而是解释问题为何无效或不连贯。如果你不知道问题的答案，也请勿提供错误的信息。<|im_end|>\n'''
    IGNORE_INDEX = -100

    def pad_tensors_to_max_length(input_tensor, max_length, pad_token_id, padding_side="right"):
        padded_tensor = pad_token_id * torch.ones((max_length,), dtype=input_tensor.dtype, device=input_tensor.device)
        try:
            if padding_side == "left":
                padded_tensor[-input_tensor.shape[0]:] = input_tensor
            else:
                padded_tensor[:input_tensor.shape[0]] = input_tensor
        except Exception as e:
            print('padding input empty:', e)
        return padded_tensor

    def _masklabel(prompt, ans):
        input_ids = tokenizer.encode(prompt, add_special_tokens=False)
        labels = [IGNORE_INDEX] * len(input_ids)
        ans_ids = tokenizer.encode(ans, add_special_tokens=False)
        input_ids += ans_ids
        labels += copy.deepcopy(ans_ids)
        input_ids = input_ids[:sft_config.max_length-1]
        labels = labels[:sft_config.max_length-1]
        attention_mask = [1] * len(input_ids)

        input_ids = torch.LongTensor(input_ids)
        attention_mask = torch.LongTensor(attention_mask)
        labels = torch.LongTensor(labels)

        out = {
            "input_ids": pad_tensors_to_max_length(input_ids, sft_config.max_length, tokenizer.pad_token_id, tokenizer.padding_side),
            "attention_mask": pad_tensors_to_max_length(attention_mask, sft_config.max_length, tokenizer.pad_token_id, tokenizer.padding_side),
            "labels": pad_tensors_to_max_length(labels, sft_config.max_length, IGNORE_INDEX, tokenizer.padding_side)
        }
        return out["input_ids"], out["attention_mask"], out["labels"]

    def process_distill_dataset(record):
        user_query = record["prompt"]
        if "prompt" not in record:
            print(f"Missing 'prompt' in record: {record}")
        if "responses" not in record or len(record["responses"]) == 0:
            print(f"Missing or empty 'responses' in record: {record}")
        sys_prompt = default_sys_prompt
        prompt = sys_prompt + '<|im_start|>user\n' + user_query.strip() + '<|im_end|>\n<|im_start|>assistant\n'
        # answer = record["responses"][0].strip() + '<|im_end|>'
        answer = record["responses"][0].strip() + ('<|im_end|>' if '<\\think>' in record["responses"][0] else '')
        chosen = prompt + answer
        chosen_input_ids, chosen_attention_mask, chosen_labels = _masklabel(prompt, answer)
        prompt_input_ids, prompt_attention_mask, _ = _masklabel(prompt, "")

        chosen_response_only = answer

        new_ret = {
            "chosen_input_ids": chosen_input_ids,
            "input_ids_chosen": chosen_input_ids,
            "chosen_attention_mask": chosen_attention_mask,
            "attention_mask_chosen": chosen_attention_mask,
            "chosen_labels": chosen_labels,
            "prompt_input_ids": prompt_input_ids,
            "prompt_attention_mask": prompt_attention_mask,
            "prompt": prompt,
            "chosen": chosen,
            "chosen_response_only": chosen_response_only,
        }
        return new_ret

    with training_args.main_process_first(desc="Process distill dataset"):
        return raw_datasets.map(
            process_distill_dataset,
            batched=False,
            num_proc=sft_config.preprocess_num_workers,
            remove_columns=raw_datasets["train"].column_names,
            desc="Process distill dataset"
        )
    

def create_dpo_dataset(training_args, sft_config, tokenizer):
    # pdb.set_trace()
    train_file_path = sft_config.train_files
    validate_file_path = sft_config.val_files
        
    raw_datasets = datasets.load_dataset("json", data_files={'train': train_file_path,
                                                             'validation': validate_file_path})
    
    default_sys_prompt = '''<|im_start|>system 你是一个名为"南北阁"的人工智能助手，正在与人类用户进行交谈。你的目标是以最有帮助和最逻辑的方式回答问题，同时确保内容的安全性。你的回答中不应包含任何有害、政治化、宗教化、不道德、种族主义、非法的内容。请确保你的回答不带有社会偏见，符合社会主义价值观。如果遇到的问题无意义或事实上不连贯，请不要回答错误的内容，而是解释问题为何无效或不连贯。如果你不知道问题的答案，也请勿提供错误的信息。<|im_end|>\n'''
    IGNORE_INDEX = -100
    def pad_tensors_to_max_length(input_tensor, max_length, pad_token_id, padding_side="right"):
        padded_tensor = pad_token_id * torch.ones((max_length,), dtype=input_tensor.dtype, device=input_tensor.device)
        try:
            if padding_side == "left":
                padded_tensor[-input_tensor.shape[0]:] = input_tensor
            else:
                padded_tensor[:input_tensor.shape[0]] = input_tensor
        except:
            print('padding input empty')
        return padded_tensor

    def _masklabel(prompt, ans):
        input_ids = tokenizer.encode(prompt, add_special_tokens=False)
        labels = [IGNORE_INDEX] * len(input_ids)
        ans_ids = tokenizer.encode(ans, add_special_tokens=False)
        input_ids += ans_ids
        labels += copy.deepcopy(ans_ids)
        
        # labels[-1]= IGNORE_INDEX  #为了特殊token不算loss
        input_ids = input_ids[:sft_config.max_length-1] # 右截断
        labels = labels[:sft_config.max_length-1]
        attention_mask = [1] * len(input_ids)

        input_ids = torch.LongTensor(input_ids)
        attention_mask = torch.LongTensor(attention_mask)
        labels = torch.LongTensor(labels)
        
        chosen_token = {
            "input_ids": pad_tensors_to_max_length(input_ids, sft_config.max_length, tokenizer.pad_token_id, tokenizer.padding_side),
            "attention_mask": pad_tensors_to_max_length(attention_mask, sft_config.max_length, tokenizer.pad_token_id, tokenizer.padding_side), #这里tokenizer.pad_token_id有点危险，如果不是0，很容易变成不被mask了；
            "labels": pad_tensors_to_max_length(labels, sft_config.max_length, IGNORE_INDEX, tokenizer.padding_side)
        }
        return chosen_token["input_ids"], chosen_token["attention_mask"], chosen_token["labels"]

    def _masklabel_reject(prompt, ans):
        input_ids = tokenizer.encode(prompt, add_special_tokens=False)
        labels = [IGNORE_INDEX] * len(input_ids)
        ans_ids = tokenizer.encode(ans, add_special_tokens=False)
        input_ids += ans_ids
        labels += copy.deepcopy(ans_ids)
        
        # labels[-1]= IGNORE_INDEX  #为了特殊token不算loss
        input_ids = input_ids[:sft_config.max_length-1] # 右截断
        labels = labels[:sft_config.max_length-1]
        attention_mask = [1] * len(input_ids)

        input_ids = torch.LongTensor(input_ids)
        attention_mask = torch.LongTensor(attention_mask)
        labels = torch.LongTensor(labels)
        
        chosen_token = {
            "input_ids": pad_tensors_to_max_length(input_ids, sft_config.max_length, tokenizer.pad_token_id, tokenizer.padding_side),
            "attention_mask": pad_tensors_to_max_length(attention_mask, sft_config.max_length, tokenizer.pad_token_id, tokenizer.padding_side),
            "labels": pad_tensors_to_max_length(labels, sft_config.max_length, IGNORE_INDEX, tokenizer.padding_side)
        }
        return chosen_token["input_ids"], chosen_token["attention_mask"], chosen_token["labels"]

    def process_dpo_dataset(record):
        # pdb.set_trace()
        if "STEPDPO" in record['id'] or "pts" in record['id']:
            ret = {
            "prompt": record['conversations'][0]['value'], 
            "chosen": record['conversations'][1]['value'] ,
            "rejected": record['conversations_reject'][1]['value'] 
        }
        
        else:
            if record['conversations'][0]['value'].startswith("<|im_start|>system"):
                dpo_prompt = record['conversations'][0]['value'].strip() + '\n\n'
            elif record['conversations'][0]['value'].startswith("<|im_start|>user") :
                dpo_prompt = default_sys_prompt  + record['conversations'][0]['value'].strip()
            else:
                dpo_prompt = default_sys_prompt + '<|im_start|>user\n' + record['conversations'][0]['value'].strip() + '<|im_end|>\n'
            dpo_prompt = dpo_prompt.replace('<\s>' ,'<|im_end|>').replace('### Human: ','<|im_start|>user').replace( '### Assistant: ','<|im_start|>assistant').replace( '### System: ','<|im_start|>system').replace( '<|eot_id|>','<|im_end|>')
            ret = {
                "prompt": dpo_prompt + random.choice(['','\n']) + '<|im_start|>assistant\n' , 
                "chosen": record['conversations'][1]['value'].strip() + '<|im_end|>',
                "rejected": record['conversations_reject'][1]['value'].strip()  + '<|im_end|>',
            }
       
        # 下面的生成并没有实际上被使用，所以这里直接注释掉；
        #chosen_input_ids, chosen_attention_mask, chosen_labels = _masklabel(ret['prompt'], ret['chosen'])
        #rejected_input_ids, rejected_attention_mask, rejected_labels = _masklabel_reject(ret['prompt'], ret['rejected'])
        #prompt_input_ids, prompt_attention_mask, _ = _masklabel(ret['prompt'], "")
        prompt = ret['prompt']
        chosen = ret['prompt'] + ret['chosen']
        rejected = ret['prompt'] + ret['rejected']
        chosen_response_only = ret['chosen']
        rejected_response_only = ret['rejected']

        new_ret = {
            '''
            "chosen_input_ids": chosen_input_ids,
            "input_ids_chosen": chosen_input_ids,
            "chosen_attention_mask": chosen_attention_mask,
            "attention_mask_chosen": chosen_attention_mask,
            "chosen_labels": chosen_labels,
            "rejected_input_ids": rejected_input_ids,
            "input_ids_rejected": rejected_input_ids,
            "rejected_attention_mask": rejected_attention_mask,
            "attention_mask_rejected": rejected_attention_mask,
            "rejected_labels": rejected_labels,
            "prompt_input_ids": prompt_input_ids,
            "prompt_attention_mask": prompt_attention_mask,
            '''
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "chosen_response_only": chosen_response_only,
            "rejected_response_only": rejected_response_only,
        }
        if 'margin' in record:
            new_ret['margin'] = record['margin']
        return new_ret

    if sft_config.margin > 0.0:
        margin = sft_config.margin
        raw_datasets = raw_datasets.map(lambda x: {'margin': margin, **x})

    #with training_args.main_process_first(desc="Process dpo dataset"):
    
    return raw_datasets.map(
        process_dpo_dataset,
        batched=False,
        num_proc=sft_config.preprocess_num_workers,
        remove_columns=raw_datasets["train"].column_names, #这个地方没有传入column_names的内容？
        desc="Process dpo dataset"
    )
    # # 单线程
    # for item in raw_datasets['train']:
    #     process_dpo_dataset(item)

class MyDPODataCollator:
    r"""
    DPO DataCollator class that pads the inputs to the maximum length of the batch.
    Args:
        tokenizer (`PreTrainedTokenizerBase`):
            The tokenizer used for encoding the data.
        model (Optional[`PreTrainedModel`]):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*.
        padding (`Union[bool, str, `PaddingStrategy`]`, `optional`, defaults to `True`):
            padding_strategy to pass to the tokenizer.
        max_length (`Optional[int]`, `optional`, defaults to `None`):
            The maximum length of the sequence to be processed.
        max_prompt_length (`Optional[int]`, `optional`, defaults to `None`):
            The maximum length of the prompt to be processed.
        label_pad_token_id (`int`, defaults to -100):
            The label used for masking.
        padding_value (`int`, defaults to 0):
            The value used for padding.
        is_encoder_decoder (`Optional[bool]`, `optional`, defaults to `None`):
            Whether or not you model has an encoder_decoder architecture.
        max_target_length (`Optional[int]`, `optional`, defaults to `None`):
            The maximum length of the target to be processed. Only useful for encoder-decoder architectures.
        truncation_mode: (`str`, defaults to "keep_end"):
            The truncation mode to use when truncating the prompt.
    """
    def collate(self, batch):
        # first, pad everything to the same length
        padded_batch = {}
        for k in batch[0].keys():
            # print(k)
            if k.endswith("_input_ids") or k.endswith("_attention_mask") or k.endswith("_labels"):
                # print([ex[k] for ex in batch])
                padded_batch[k] = torch.LongTensor([ex[k] for ex in batch])
                # print(padded_batch[k].shape)
            else:
                padded_batch[k] = [ex[k] for ex in batch]

        return padded_batch

    def __call__(self, features):
        batch = features
        padded_batch = {}
        for k in batch[0].keys():
            # print(k)
            if k.endswith("_input_ids") or k.endswith("_attention_mask") or k.endswith("_labels"):
                # print([ex[k] for ex in batch])
                padded_batch[k] = torch.LongTensor([ex[k] for ex in batch])
                # print(padded_batch[k].shape)
            else:
                padded_batch[k] = [ex[k] for ex in batch]

        return padded_batch

@dataclass
class MyDPODataCollator2:
    r"""
    DPO DataCollator class that pads the inputs to the maximum length of the batch.
    Args:
        tokenizer (`PreTrainedTokenizerBase`):
            The tokenizer used for encoding the data.
        model (Optional[`PreTrainedModel`]):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*.
        padding (`Union[bool, str, `PaddingStrategy`]`, `optional`, defaults to `True`):
            padding_strategy to pass to the tokenizer.
        max_length (`Optional[int]`, `optional`, defaults to `None`):
            The maximum length of the sequence to be processed.
        max_prompt_length (`Optional[int]`, `optional`, defaults to `None`):
            The maximum length of the prompt to be processed.
        label_pad_token_id (`int`, defaults to -100):
            The label used for masking.
        padding_value (`int`, defaults to 0):
            The value used for padding.
        is_encoder_decoder (`Optional[bool]`, `optional`, defaults to `None`):
            Whether or not you model has an encoder_decoder architecture.
        max_target_length (`Optional[int]`, `optional`, defaults to `None`):
            The maximum length of the target to be processed. Only useful for encoder-decoder architectures.
        truncation_mode: (`str`, defaults to "keep_end"):
            The truncation mode to use when truncating the prompt.
    """
    tokenizer: PreTrainedTokenizerBase
    model: Optional[PreTrainedModel] = None
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_prompt_length: Optional[int] = None
    label_pad_token_id: int = -100
    padding_value: int = 0
    truncation_mode: str = "keep_end"
    is_encoder_decoder: Optional[bool] = False
    max_target_length: Optional[int] = None

    def tokenize_batch_element(
        self,
        prompt: str,
        chosen: str,
        rejected: str,
    ) -> Dict:
        """Tokenize a single batch element.

        At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
            in case the prompt + chosen or prompt + rejected responses is/are too long. First
            we truncate the prompt; if we're still too long, we truncate the chosen/rejected.

        We also create the labels for the chosen/rejected responses, which are of length equal to
            the sum of the length of the prompt and the chosen/rejected response, with
            label_pad_token_id  for the prompt tokens.
        """
        batch = {}

        if not self.is_encoder_decoder:
            chosen_tokens = self.tokenizer(chosen, add_special_tokens=False)
            rejected_tokens = self.tokenizer(rejected, add_special_tokens=False)
            prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)

            eos_token_id = self.tokenizer.eos_token_id
            # Get indices in list prompt_tokens["input_ids"] that equals the EOS token (often 0)
            eos_indices_prompt = [i for i, x in enumerate(prompt_tokens["input_ids"]) if x == eos_token_id]
            # attention mask these indices to eos_token_id
            new_attention_mask = [
                0 if i in eos_indices_prompt else p for i, p in enumerate(prompt_tokens["attention_mask"])
            ]
            prompt_tokens["attention_mask"] = new_attention_mask

            # do the same for chosen and rejected
            eos_indices_chosen = [i for i, x in enumerate(chosen_tokens["input_ids"]) if x == eos_token_id]
            new_attention_mask_c = [
                0 if i in eos_indices_chosen else p for i, p in enumerate(chosen_tokens["attention_mask"])
            ]
            chosen_tokens["attention_mask"] = new_attention_mask_c

            eos_indices_rejected = [i for i, x in enumerate(rejected_tokens["input_ids"]) if x == eos_token_id]
            new_attention_mask_r = [
                0 if i in eos_indices_rejected else p for i, p in enumerate(rejected_tokens["attention_mask"])
            ]
            rejected_tokens["attention_mask"] = new_attention_mask_r

            # add EOS token to end of prompt
            chosen_tokens["input_ids"].append(self.tokenizer.eos_token_id)
            chosen_tokens["attention_mask"].append(1)

            rejected_tokens["input_ids"].append(self.tokenizer.eos_token_id)
            rejected_tokens["attention_mask"].append(1)

            longer_response_length = max(len(chosen_tokens["input_ids"]), len(rejected_tokens["input_ids"]))

            # if combined sequence is too long, truncate the prompt
            if len(prompt_tokens["input_ids"]) + longer_response_length > self.max_length:
                if self.truncation_mode == "keep_start":
                    prompt_tokens = {k: v[: self.max_prompt_length] for k, v in prompt_tokens.items()}
                elif self.truncation_mode == "keep_end":
                    prompt_tokens = {k: v[-self.max_prompt_length :] for k, v in prompt_tokens.items()}
                else:
                    raise ValueError(f"Unknown truncation mode: {self.truncation_mode}")

            # if that's still too long, truncate the response
            if len(prompt_tokens["input_ids"]) + longer_response_length > self.max_length:
                chosen_tokens = {k: v[: self.max_length - self.max_prompt_length] for k, v in chosen_tokens.items()}
                rejected_tokens = {
                    k: v[: self.max_length - self.max_prompt_length] for k, v in rejected_tokens.items()
                }

            # Create labels
            chosen_sequence_tokens = {k: prompt_tokens[k] + chosen_tokens[k] for k in chosen_tokens}
            rejected_sequence_tokens = {k: prompt_tokens[k] + rejected_tokens[k] for k in rejected_tokens}
            chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]
            chosen_sequence_tokens["labels"][: len(prompt_tokens["input_ids"])] = [self.label_pad_token_id] * len(
                prompt_tokens["input_ids"]
            )
            rejected_sequence_tokens["labels"] = rejected_sequence_tokens["input_ids"][:]
            rejected_sequence_tokens["labels"][: len(prompt_tokens["input_ids"])] = [self.label_pad_token_id] * len(
                prompt_tokens["input_ids"]
            )

            for k, toks in {
                "chosen": chosen_sequence_tokens,
                "rejected": rejected_sequence_tokens,
                "prompt": prompt_tokens,
            }.items():
                for type_key, tokens in toks.items():
                    if type_key == "token_type_ids":
                        continue
                    batch[f"{k}_{type_key}"] = tokens

        else:
            chosen_tokens = self.tokenizer(
                chosen, truncation=True, max_length=self.max_target_length, add_special_tokens=True
            )
            rejected_tokens = self.tokenizer(
                rejected, truncation=True, max_length=self.max_target_length, add_special_tokens=True
            )
            prompt_tokens = self.tokenizer(
                prompt, truncation=True, max_length=self.max_prompt_length, add_special_tokens=True
            )

            batch["chosen_labels"] = chosen_tokens["input_ids"]
            batch["rejected_labels"] = rejected_tokens["input_ids"]
            batch["prompt_input_ids"] = prompt_tokens["input_ids"]
            batch["prompt_attention_mask"] = prompt_tokens["attention_mask"]

            if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
                batch["rejected_decoder_input_ids"] = self.model.prepare_decoder_input_ids_from_labels(
                    labels=batch["rejected_labels"]
                )
                batch["chosen_decoder_input_ids"] = self.model.prepare_decoder_input_ids_from_labels(
                    labels=batch["chosen_labels"]
                )

        batch["prompt"] = prompt
        batch["chosen"] = prompt + chosen
        batch["rejected"] = prompt + rejected
        batch["chosen_response_only"] = chosen
        batch["rejected_response_only"] = rejected

        return batch

    def collate(self, batch):
        # first, pad everything to the same length
        padded_batch = {}
        for k in batch[0].keys():
            if k.endswith("_input_ids") or k.endswith("_attention_mask") or k.endswith("_labels"):
                if self.is_encoder_decoder:
                    to_pad = [torch.LongTensor(ex[k]) for ex in batch]

                    if (k.startswith("prompt")) and (k.endswith("input_ids")):
                        padding_value = self.tokenizer.pad_token_id
                    elif k.endswith("_attention_mask"):
                        padding_value = 0
                    elif (k.startswith("chosen")) or (k.startswith("rejected")) or ("decoder" in k):
                        padding_value = self.label_pad_token_id
                    else:
                        raise ValueError(f"Unexpected key in batch '{k}'")
                    padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                else:
                    # adapted from https://stackoverflow.com/questions/73256206
                    if "prompt" in k:
                        to_pad = [torch.LongTensor(ex[k][::-1]) for ex in batch]
                    else:
                        to_pad = [torch.LongTensor(ex[k]) for ex in batch]
                    if k.endswith("_input_ids"):
                        padding_value = self.tokenizer.pad_token_id
                    elif k.endswith("_labels"):
                        padding_value = self.label_pad_token_id
                    elif k.endswith("_attention_mask"):
                        padding_value = self.padding_value
                    else:
                        raise ValueError(f"Unexpected key in batch '{k}'")

                    padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                    # for the prompt, flip back so padding is on left side
                    if "prompt" in k:
                        padded_batch[k] = padded_batch[k].flip(dims=[1])
            else:
                padded_batch[k] = [ex[k] for ex in batch]

        return padded_batch
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # batch = features
        # padded_batch = {}
        # for k in batch[0].keys():
        #     # print(k)
        #     if k.endswith("_input_ids") or k.endswith("_attention_mask") or k.endswith("_labels"):
        #         # print([ex[k] for ex in batch])
        #         padded_batch[k] = torch.LongTensor([ex[k] for ex in batch])
        #         # print(padded_batch[k].shape)
        #     else:
        #         padded_batch[k] = [ex[k] for ex in batch]

        # return padded_batch
        # return features
        tokenized_batch = []

        for feature in features:
            prompt = feature["prompt"]
            chosen = feature["chosen"]
            rejected = feature["rejected"]

            batch_element = self.tokenize_batch_element(prompt, chosen, rejected)
            tokenized_batch.append(batch_element)

        # return collated batch
        return self.collate(tokenized_batch)

def get_hh(path: str, sanity_check: bool = False, silent: bool = False, cache_dir: str = None) -> Dataset:
    """Load the Anthropic Helpful-Harmless dataset from Hugging Face and convert it to the necessary format.

    The dataset is converted to a dictionary with the following structure:
    {
        'prompt': List[str],
        'chosen': List[str],
        'rejected': List[str],
    }

    Prompts should be structured as follows:
      \n\nHuman: <prompt>\n\nAssistant:
    Multiple turns are allowed, but the prompt should always start with \n\nHuman: and end with \n\nAssistant:.
    """        
    dataset = load_dataset("json", data_files=path)["train"]
    # dataset = load_dataset(path)
    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 1000)))

    def extract_anthropic_prompt(prompt_and_response):
        """Extract the anthropic prompt from a prompt and response pair."""
        search_term = "\n\nAssistant:"
        search_term_idx = prompt_and_response.rfind(search_term)
        assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
        return prompt_and_response[: search_term_idx + len(search_term)]

    def split_prompt_and_responses(sample) -> Dict[str, str]:
        prompt = extract_anthropic_prompt(sample["chosen"])
        return {
            "prompt": prompt,
            "chosen": sample["chosen"][len(prompt) :],
            "rejected": sample["rejected"][len(prompt) :],
        }
    # pdb.set_trace()
    return dataset.map(split_prompt_and_responses)

if __name__ == "__main__":
    # pdb.set_trace()
    train_dataset = get_hh("/code/zhangwenxuan/datasets/Anthropic/hh-rlhf-json/test")