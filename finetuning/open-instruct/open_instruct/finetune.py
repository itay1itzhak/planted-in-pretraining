#!/usr/bin/env python
# coding=utf-8

import argparse
import logging
import math
import os
import random
import shutil
import datasets
from datetime import timedelta
import numpy as np
import torch
from functools import partial
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed, InitProcessGroupKwargs
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import deepspeed

try:
    import hf_olmo
except ImportError:
    pass

import transformers
import torch
from datasets import Dataset
import datasets as ds
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    LlamaTokenizerFast,
    SchedulerType,
    DataCollatorForSeq2Seq,
    get_scheduler,
    GPTNeoXTokenizerFast,
    GPT2Tokenizer,
    OPTForCausalLM,
    BitsAndBytesConfig,
    T5Tokenizer,
    T5ForConditionalGeneration,
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

logger = get_logger(__name__)

try:
    from hf_olmo import OLMoTokenizerFast
except ImportError:
    try:
        logger.warning("OLMo not installed. Ignore if using a different model.")
    except:
        print("OLMo not installed. Ignore if using a different model.")



def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a causal language modeling task"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default=None,
        help="A csv or a json file containing the training data.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--model_revision",
        help="""If given, specifies a model revision (for HuggingFace models). This will 
        be applied to both the `model_name_or_path` and `config_name` args.""",
        default="main",
        required=False,
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="If passed, will use LORA (low-rank parameter-efficient training) to train the model.",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=64,
        help="The rank of lora.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=16,
        help="The alpha parameter of lora.",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="The dropout rate of lora modules.",
    )
    parser.add_argument(
        "--use_flash_attn",
        action="store_true",
        help="If passed, will use flash attention to train the model.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_revision",
        help="""Specifies a revision for the tokenizer. If not given, defaults
             to the value of the `model_revision` arg. In most cases, the tokenizer
             revision should be the same as the model revision and this flag shouldn't
             be needed.""",
        default=None,
        required=False,
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="The maximum total sequence length (prompt+completion) of each training example.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay to use."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0,
        help="Ratio of total training steps used for warmup.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Where to store the final model."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="The number of processes to use for the dataloader.",
    )
    parser.add_argument(
        "--dataloader_prefetch_factor",
        type=int,
        default=0,
        help="The number of batches to prefetch in the dataloader.",
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=None,
        help="Log the training loss and learning rate every logging_steps steps.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder. Last checkpoint can be used by passing 'last'.",
    )

    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        help=(
            "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
            "If passed, LLM loading time and RAM consumption will be benefited."
        ),
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help=("Turn on gradient checkpointing. Saves memory but slows training."),
    )
    parser.add_argument(
        "--use_qlora",
        action="store_true",
        help=(
            "Use qLoRA training - main thing is initialising model in quantised form. Not compatible with deepspeed."
        ),
    )
    parser.add_argument(
        "--clip_grad_norm",
        type=float,
        default=-1,
        help="Clip gradient norm. Not compatible with deepspeed (use deepspeed config instead).",
    )
    parser.add_argument(
        "--use_8bit_optimizer",
        action="store_true",
        help="Use 8bit optimizer from bitsandbytes. Not compatible with deepspeed (use deepspeed config instead).",
    )
    parser.add_argument(
        "--add_bos",
        action="store_true",
        help="Forcibly add bos token to the beginning of the input sequence. Use only when tokenizer does not add bos token by default (e.g., olmo).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Timeout for the training process. Useful if tokenization process is long. Default is 1800 seconds (30 minutes).",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Trust remote code when loading pretrained models and tokenizers. Use only when you trust the remote code.",
    )
    parser.add_argument(
        "--reduce_loss",
        default="mean",
        choices=["mean", "sum"],
        help="How to reduce loss over tokens. Default is mean, but using sum can improve chat model performance.",
    )
    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        help="Run evaluation during training at each checkpointing_steps.",
    )
    parser.add_argument(
        "--remove_previous_checkpoint",
        action="store_true",
        help="Remove previous checkpoint folder if it exists.",
    )
    parser.add_argument(
        "--cache_dir_dataset",
        type=str,
        default=None,
        help="Directory to cache datasets.",
    )
    parser.add_argument(
        "--cache_dir_model",
        type=str,
        default=None,
        help="Directory to cache models.",
    )
    parser.add_argument(
        "--debug_small_scale",
        action="store_true",
        help="Debug with a small scale of data.",
    )

    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None and args.train_file is None:
        raise ValueError("Need either a dataset name or a training file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in [
                "json",
                "jsonl",
            ], "`train_file` should be a json/jsonl file."
    return args


def encode_with_prompt_completion_format(
    example, tokenizer, max_seq_length, add_bos=False, add_eos=True
):
    """
    Here we assume each example has 'prompt' and 'completion' fields.
    We concatenate prompt and completion and tokenize them together because otherwise prompt will be padded/trancated
    and it doesn't make sense to follow directly with the completion.
    """
    # if prompt doesn't end with space and completion doesn't start with space, add space
    if not example["prompt"].endswith((" ", "\n", "\t")) and not example[
        "completion"
    ].startswith((" ", "\n", "\t")):
        example_text = example["prompt"] + " " + example["completion"]
    else:
        example_text = example["prompt"] + example["completion"]
    if add_eos:
        example_text = example_text + tokenizer.eos_token
    if add_bos:
        example_text = tokenizer.bos_token + example_text
    tokenized_example = tokenizer(
        example_text, return_tensors="pt", max_length=max_seq_length, truncation=True
    )
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()
    tokenized_prompt = tokenizer(
        example["prompt"],
        return_tensors="pt",
        max_length=max_seq_length,
        truncation=True,
    )
    # mask the prompt part for avoiding loss
    labels[:, : tokenized_prompt.input_ids.shape[1]] = -100
    attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids.flatten(),
        "labels": labels.flatten(),
        "attention_mask": attention_mask.flatten(),
    }


def encode_with_messages_format(example, tokenizer, max_seq_length, add_bos=False):
    """
    Here we assume each example has a 'messages' field Each message is a dict with 'role' and 'content' fields.
    We concatenate all messages with the roles as delimiters and tokenize them together.
    """
    messages = example["messages"]
    if len(messages) == 0:
        raise ValueError("messages field is empty.")

    def _concat_messages(messages):
        message_text = ""
        for message in messages:
            if message["role"] == "system":
                message_text += "<|system|>\n" + message["content"].strip() + "\n"
            elif message["role"] == "user":
                message_text += "<|user|>\n" + message["content"].strip() + "\n"
            elif message["role"] == "assistant":
                message_text += (
                    "<|assistant|>\n"
                    + message["content"].strip()
                    + tokenizer.eos_token
                    + "\n"
                )
            else:
                raise ValueError("Invalid role: {}".format(message["role"]))
        return message_text

    example_text = _concat_messages(messages).strip()
    if add_bos:
        example_text = tokenizer.bos_token + example_text
    tokenized_example = tokenizer(
        example_text, return_tensors="pt", max_length=max_seq_length, truncation=True
    )
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()

    # mask the non-assistant part for avoiding loss
    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = tokenizer(
                    _concat_messages(messages[:message_idx]),
                    return_tensors="pt",
                    max_length=max_seq_length,
                    truncation=True,
                ).input_ids.shape[1]
            if (
                message_idx < len(messages) - 1
                and messages[message_idx + 1]["role"] == "assistant"
            ):
                # here we also ignore the role of the assistant
                messages_so_far = (
                    _concat_messages(messages[: message_idx + 1]) + "<|assistant|>\n"
                )
            else:
                messages_so_far = _concat_messages(messages[: message_idx + 1])
            message_end_idx = tokenizer(
                messages_so_far,
                return_tensors="pt",
                max_length=max_seq_length,
                truncation=True,
            ).input_ids.shape[1]
            labels[:, message_start_idx:message_end_idx] = -100

            if message_end_idx >= max_seq_length:
                break

    attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids.flatten(),
        "labels": labels.flatten(),
        "attention_mask": attention_mask.flatten(),
    }


def merge_datasets_with_weights(datasets, N):
    from collections import Counter

    def sample_weighted(dataset, num_samples, weighted_column=None):
        if weighted_column is not None:
            # Calculate the ratio of each unique value in the weighted column
            value_counts = Counter(dataset[weighted_column])
            total_count = sum(value_counts.values())
            weights = {val: count / total_count for val, count in value_counts.items()}

            # Sample according to the weighted_column
            sampled_indices = []
            for val, weight in weights.items():
                val_indices = [
                    i for i, x in enumerate(dataset[weighted_column]) if x == val
                ]
                num_val_samples = round(weight * num_samples)
                sampled_indices.extend(
                    random.sample(val_indices, min(num_val_samples, len(val_indices)))
                )

            return dataset.select(sampled_indices)
        else:
            indices = random.sample(range(len(dataset)), num_samples)
            return dataset.select(indices)

    # Calculate total number of examples in all datasets
    total_examples = sum(len(dataset) for dataset in datasets)

    # Calculate the weight for each dataset
    weights = [len(dataset) / total_examples for dataset in datasets]

    # Calculate number of samples needed from each dataset
    samples_per_dataset = [round(weight * N) for weight in weights]

    # Sample from each dataset while maintaining inner partitioning
    sampled_datasets = []
    for i, dataset in enumerate(datasets):
        print(f"Sampling {samples_per_dataset[i]} examples from dataset {i}...")
        task_sampled = sample_weighted(
            dataset, samples_per_dataset[i], weighted_column="task_name"
        )
        full_sampled = sample_weighted(
            task_sampled, len(task_sampled), weighted_column="template_type"
        )
        sampled_datasets.extend(full_sampled)

    # Combine all sampled datasets
    combined_dataset = Dataset.from_dict(
        {key: [d[key] for d in sampled_datasets] for key in sampled_datasets[0].keys()}
    )

    return combined_dataset


def load_flan_dataset(
    cache_dir="huggingface/datasets", dataset_name="itay1itzhak/flan_2022_350k"
):
    # Load from huggingface
    try:   
        combined_dataset = datasets.load_dataset(dataset_name, split="train", cache_dir=cache_dir)
        print(f"Loaded dataset from {dataset_name}")
        return combined_dataset
    except Exception as e:
        print(f"Error loading dataset{dataset_name} from huggingface: {e}")
        print("Recreating dataset from scratch...")
        from scripts.create_flan_2022_350k_subsample import load_flan_dataset
        combined_dataset = load_flan_dataset(cache_dir=cache_dir)
        return combined_dataset


def get_prompt_completion_format(example, tokenizer):
    input_ids = example["input_ids"]
    label_mask = example["label_mask"]

    prompt_text = []
    completion_text = []

    for i in range(len(input_ids)):
        if label_mask[i] == False:
            prompt_text.append(input_ids[i])
        else:
            completion_text.append(input_ids[i])

    # Decode lists of token IDs into strings
    prompt_text = tokenizer.decode(prompt_text).strip()
    completion_text = tokenizer.decode(completion_text).strip()

    return prompt_text, completion_text


def encode_flan_for_olmo(example, tokenizer, max_seq_length, add_bos=False):
    # Get the prompt and completion text
    prompt_text, completion_text = get_prompt_completion_format(example, tokenizer)

    # Create example dictionary
    example = {"prompt": prompt_text, "completion": completion_text}

    # Use the encode_with_prompt_completion_format function to preprocess
    processed_example = encode_with_prompt_completion_format(
        example, tokenizer, max_seq_length, add_bos=add_bos
    )

    return processed_example


def encode_flan_for_t5(example, tokenizer, max_seq_length, add_bos=False):
    # Get the prompt and completion text
    # prompt_text, completion_text = get_prompt_completion_format(example, tokenizer)

    # Create example dictionary
    # example = {"prompt": prompt_text, "completion": completion_text}

    # Use the encode_with_prompt_completion_format function to preprocess
    # processed_example = encode_with_prompt_completion_format(
    #     example, tokenizer, max_seq_length, add_bos=add_bos, add_eos=False
    # )

    # tokenize only inputs
    processed_example = tokenizer(
        example["prompt"],
        return_tensors="pt",
        max_length=max_seq_length,
        truncation=True,
        padding=True,
    )
    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(
        text_target=example["completion"],
        return_tensors="pt",
        max_length=256,  # decoder max_seq_length,
        truncation=True,
    )  # ["input_ids"]

    # Replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore padding in the loss
    # processed_example["labels"] = [
    #     [(l if l != tokenizer.pad_token_id else -100) for l in label]
    #     for label in labels["input_ids"]
    # ]
    # processed_example["labels"] = [
    #     [(l.item() if l != tokenizer.pad_token_id else -100) for l in label]
    #     for label in labels["input_ids"]
    # ]
    # processed_example["labels"] = labels["input_ids"]

    # this is the real code
    lm_labels = labels["input_ids"]  # .clone().detach()
    lm_labels[labels["input_ids"] == tokenizer.pad_token_id] = -100
    processed_example["labels"] = lm_labels.flatten()
    processed_example["input_ids"] = processed_example["input_ids"].flatten()
    processed_example["attention_mask"] = processed_example["attention_mask"].flatten()

    return processed_example


def encode_tulu_for_t5_with_messages_format(
    example, tokenizer, max_seq_length, add_bos=False
):
    """
    Here we assume each example has a 'messages' field Each message is a dict with 'role' and 'content' fields.
    We concatenate all messages except the last one with the roles as delimiters and tokenize them together.
    The last message is the target completion.
    """
    messages = example["messages"]
    if len(messages) == 0:
        raise ValueError("messages field is empty.")

    def _concat_messages(messages):
        message_text = ""
        for message in messages:
            if message["role"] == "system":
                message_text += "<|system|>\n" + message["content"].strip() + "\n"
            elif message["role"] == "user":
                message_text += "<|user|>\n" + message["content"].strip() + "\n"
            elif message["role"] == "assistant":
                message_text += (
                    "<|assistant|>\n"
                    + message["content"].strip()
                    + tokenizer.eos_token
                    + "\n"
                )
            else:
                raise ValueError("Invalid role: {}".format(message["role"]))
        return message_text

    input_messages = messages[:-1]
    target_message = messages[-1]

    example_text = _concat_messages(input_messages).strip()
    if add_bos:
        example_text = tokenizer.bos_token + example_text
    # need to add assistent token to the end of the input text so it's in the encoder
    example_text += "\n<|assistant|>\n"
    tokenized_example = tokenizer(
        example_text, return_tensors="pt", max_length=max_seq_length, truncation=True
    )
    input_ids = tokenized_example.input_ids
    # labels = input_ids.clone()
    tokenized_target = tokenizer(
        # _concat_messages([target_message]),
        target_message["content"],
        return_tensors="pt",
        max_length=256,  # decoder max_seq_length,
        truncation=True,
    )

    labels = tokenized_target.input_ids
    labels[labels == tokenizer.pad_token_id] = -100

    # mask the non-assistant part for avoiding loss
    # for message_idx, message in enumerate(input_messages):
    #     if message["role"] != "assistant":
    #         if message_idx == 0:
    #             message_start_idx = 0
    #         else:
    #             message_start_idx = tokenizer(
    #                 _concat_messages(messages[:message_idx]),
    #                 return_tensors="pt",
    #                 max_length=max_seq_length,
    #                 truncation=True,
    #             ).input_ids.shape[1]
    #         if (
    #             message_idx < len(messages) - 1
    #             and messages[message_idx + 1]["role"] == "assistant"
    #         ):
    #             # here we also ignore the role of the assistant
    #             messages_so_far = (
    #                 _concat_messages(messages[: message_idx + 1]) + "<|assistant|>\n"
    #             )
    #         else:
    #             messages_so_far = _concat_messages(messages[: message_idx + 1])
    #         message_end_idx = tokenizer(
    #             messages_so_far,
    #             return_tensors="pt",
    #             max_length=max_seq_length,
    #             truncation=True,
    #         ).input_ids.shape[1]
    #         labels[:, message_start_idx:message_end_idx] = -100

    #         if message_end_idx >= max_seq_length:
    #             break

    attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids.flatten(),
        "labels": labels.flatten(),
        "attention_mask": attention_mask.flatten(),
    }


def save_with_accelerate(
    accelerator, model, tokenizer, output_dir, args, optimizer=None, lr_scheduler=None
):
    unwrapped_model = accelerator.unwrap_model(model)
    # When doing multi-gpu training, we need to use accelerator.get_state_dict(model) to get the state_dict.
    # Otherwise, sometimes the model will be saved with only part of the parameters.
    # Also, accelerator needs to use the wrapped model to get the state_dict.
    state_dict = accelerator.get_state_dict(model)
    if args.use_lora:
        # When using lora, the unwrapped model is a PeftModel, which doesn't support the is_main_process
        # and has its own save_pretrained function for only saving lora modules.
        # We have to manually specify the is_main_process outside the save_pretrained function.
        if accelerator.is_main_process:
            # unwrapped_model.save_pretrained(output_dir, state_dict=state_dict)

            unwrapped_model.save_pretrained(
                output_dir,
                save_function=accelerator.save,
                state_dict=state_dict,
            )

            # save state_dict for lora
            # torch.save(state_dict, os.path.join(output_dir, "state_dict.pt"))
            # accelerator.save_state(output_dir)
            # Save the optimizer state
            # optimizer_path = os.path.join(output_dir, "optimizer.pt")
            # torch.save(optimizer.state_dict(), optimizer_path)

            # # Save the scheduler state
            # scheduler_path = os.path.join(output_dir, "scheduler.pt")
            # torch.save(lr_scheduler.state_dict(), scheduler_path)

        # Save the accelerator state
        accelerator_state_path = os.path.join(output_dir, "accelerator_state")
        accelerator.save_state(accelerator_state_path)

        # # Save the tokenizer
        # tokenizer.save_pretrained(output_dir)

    else:
        # don't use safetensors for saving for now
        unwrapped_model.save_pretrained(
            output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=state_dict,
            safe_serialization=False,
        )


def evaluate_model(model, tokenizer, args, accelerator, step=None, eval_dataset="mmlu"):
    """
    evaluate the model on the eval_dataset using the script from scripts/eval directory. Make sure to consider sevral process with accelerator

    """
    if eval_dataset == "mmlu":
        from eval.mmlu import run_eval_olmo
    else:
        raise ValueError("eval_dataset must be 'mmlu'")

    # get the model path
    model_path = os.path.join(args.output_dir, f"step_{step}")
    # get the output path
    output_path = os.path.join(args.output_dir, "eval_results")
    # run the eval script
    # Synchronize before running the command
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        print(f"Running evaluation on {eval_dataset} dataset...")
        # set the accelerator to eval mode
        model.eval()
        weighted_acc = run_eval_olmo.predict_mmlu(
            model,
            tokenizer,
            model_path,
            use_chat_format=True,
            save_dir=output_path,
            disable_tqdm=True,
        )
        # return the model to train mode
        model.train()
        return weighted_acc
    else:
        # Non-main processes wait for the main process to complete
        accelerator.wait_for_everyone()
        return None


def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    # if you get timeouts (e.g. due to long tokenization) increase this.
    timeout_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=args.timeout))

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        **accelerator_log_kwargs,
        kwargs_handlers=[timeout_kwargs],
    )
    # new_accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps,**accelerator_log_kwargs,kwargs_handlers=[timeout_kwargs])
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    accelerator.wait_for_everyone()

    if args.dataset_name is not None:
        if args.dataset_name == "flan_2022":
            raw_datasets = load_flan_dataset(
                args.max_seq_length, cache_dir=args.cache_dir_dataset
            )
        else:
            # Downloading and loading a dataset from the hub.
            raw_datasets = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                cache_dir=args.cache_dir_dataset,
            )
    else:
        data_files = {}
        dataset_args = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        raw_datasets = load_dataset(
            "json",
            data_files=data_files,
            **dataset_args,
        )

    # Load pretrained model and tokenizer
    if args.config_name:
        config = AutoConfig.from_pretrained(
            args.config_name,
            trust_remote_code=args.trust_remote_code,
            revision=args.model_revision,
        )
    elif args.model_name_or_path:
        print("trust_remote_code", args.trust_remote_code)
        config = AutoConfig.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=args.trust_remote_code,
            revision=args.model_revision,
        )
    else:
        raise ValueError(
            "You are instantiating a new config instance from scratch. This is not supported by this script."
        )

    tokenizer_revision = (
        args.model_revision
        if args.tokenizer_revision is None
        else args.tokenizer_revision
    )

    if tokenizer_revision != args.model_revision:
        # Warn user if tokenizer and model use different revisions; this is an unusual
        # use case.
        warning = f"""Requested tokenizer revision `{tokenizer_revision}` is different
                   from the model revision `{args.model_revision}`."""
        logger.warn(warning)

    if args.tokenizer_name:
        # if not t5 mode load normal tokenizer
        if not "t5" in args.model_name_or_path:
            tokenizer = AutoTokenizer.from_pretrained(
                args.tokenizer_name,
                trust_remote_code=args.trust_remote_code,
                use_fast=not args.use_slow_tokenizer,
                revision=tokenizer_revision,
            )
        else:  # load with legacy tokenizer false
            tokenizer = AutoTokenizer.from_pretrained(
                args.tokenizer_name,
                legacy=False,
                use_fast=not args.use_slow_tokenizer,
                revision=tokenizer_revision,
            )

    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=args.trust_remote_code,
            use_fast=not args.use_slow_tokenizer,
            revision=tokenizer_revision,
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if args.model_name_or_path:
        if args.use_qlora:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            device_index = accelerator.local_process_index
            device_map = {"": device_index}  # force data-parallel training.
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                load_in_4bit=True,
                quantization_config=bnb_config,
                device_map=device_map,
                trust_remote_code=args.trust_remote_code,
                torch_dtype=torch.bfloat16,
                use_flash_attention_2=True if args.use_flash_attn else False,
                revision=args.model_revision,
            )
        else:
            if not "t5" in args.model_name_or_path:
                model = AutoModelForCausalLM.from_pretrained(
                    args.model_name_or_path,
                    from_tf=bool(".ckpt" in args.model_name_or_path),
                    config=config,
                    trust_remote_code=args.trust_remote_code,
                    low_cpu_mem_usage=args.low_cpu_mem_usage,
                    use_flash_attention_2=True if args.use_flash_attn else False,
                    revision=args.model_revision,
                    cache_dir=args.cache_dir_model,
                )
            else:
                model = T5ForConditionalGeneration.from_pretrained(
                    args.model_name_or_path,
                    from_tf=bool(".ckpt" in args.model_name_or_path),
                    config=config,
                    revision=args.model_revision,
                    cache_dir=args.cache_dir_model,
                )

    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config)

    # no default pad token for llama!
    # here we add all special tokens again, because the default ones are not in the special_tokens_map
    if isinstance(tokenizer, LlamaTokenizer) or isinstance(
        tokenizer, LlamaTokenizerFast
    ):
        num_added_tokens = tokenizer.add_special_tokens(
            {
                "bos_token": "<s>",
                "eos_token": "</s>",
                "unk_token": "<unk>",
                "pad_token": "<pad>",
            }
        )
        assert num_added_tokens in [
            0,
            1,
        ], "LlamaTokenizer should only add one special token - the pad_token, or no tokens if pad token present."
    elif isinstance(tokenizer, GPTNeoXTokenizerFast):
        num_added_tokens = tokenizer.add_special_tokens(
            {
                "pad_token": "<pad>",
            }
        )
        assert (
            num_added_tokens == 1
        ), "GPTNeoXTokenizer should only add one special token - the pad_token."
    elif isinstance(tokenizer, GPT2Tokenizer) and isinstance(model, OPTForCausalLM):
        num_added_tokens = tokenizer.add_special_tokens({"unk_token": "<unk>"})
    elif isinstance(tokenizer, OLMoTokenizerFast):
        # only the eos for olmo, but we use it as bos
        tokenizer.bos_token = tokenizer.eos_token
        assert (
            args.add_bos
        ), "For OLMo, you must add bos token to the beginning of the input sequence."

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    # gather deepspeed to get "real" embedding size
    embeddings = model.get_input_embeddings()
    with deepspeed.zero.GatheredParameters(embeddings.weight, modifier_rank=None):
        embedding_size = embeddings.weight.shape[0]
        if len(tokenizer) > embeddings.weight.shape[0]:
            model.resize_token_embeddings(len(tokenizer))

    if args.use_lora:
        if args.use_qlora:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=args.gradient_checkpointing
            )

        logger.info("Initializing LORA model...")
        print(model)
        if "t5" in args.model_name_or_path:
            peft_config = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                inference_mode=False,
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=[
                    "q",
                    "k",
                    "v",
                    "o",
                    "wi",
                    "wo",
                ],
            )
            model = get_peft_model(model, peft_config)
            # Trying to load on different gpus
            # from accelerate import dispatch_model, infer_auto_device_map
            # from accelerate.utils import get_balanced_memory

            # max_memory = get_balanced_memory(
            #     model,
            #     max_memory=None,
            #     no_split_module_classes=["encoder", "decoder"],
            #     dtype="bfloat16",
            #     low_zero=False,
            # )

            # device_map = infer_auto_device_map(
            #     model,
            #     max_memory=max_memory,
            #     no_split_module_classes=["encoder", "decoder"],
            #     dtype="bfloat16",
            # )

            # model = dispatch_model(model, device_map=device_map)
        else:
            try:
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    r=args.lora_rank,
                    lora_alpha=args.lora_alpha,
                    lora_dropout=args.lora_dropout,
                    target_modules=[
                        # "q_proj",
                        # "o_proj",
                        # "v_proj",
                        # "k_proj",
                        # "gate_proj",
                        # "up_proj",
                        # "down_proj",
                        "att_proj",
                        "attn_out",
                        "ff_proj",
                        "ff_out",
                    ],
                )
                model = get_peft_model(model, peft_config)
            except ValueError as e:
                logger.error(
                    f"Error initializing LORA with modules: att_proj, attn_out, ff_proj, ff_out, trying again with default modules."
                )
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    r=args.lora_rank,
                    lora_alpha=args.lora_alpha,
                    lora_dropout=args.lora_dropout,
                    target_modules=[
                        "q_proj",
                        "o_proj",
                        "v_proj",
                        "k_proj",
                        "gate_proj",
                        "up_proj",
                        "down_proj",
                        # "att_proj",
                        # "attn_out",
                        # "ff_proj",
                        # "ff_out",
                    ],
                )
                model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    elif args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Preprocessing the datasets.
    if "t5" in args.model_name_or_path:  # flan dataset
        # if model is t5
        if "flan" in args.dataset_name:
            encode_function = partial(
                encode_flan_for_t5,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                add_bos=args.add_bos,
            )
        else:
            encode_function = partial(
                encode_tulu_for_t5_with_messages_format,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                add_bos=args.add_bos,
            )
    elif "input_ids" in raw_datasets:  # flan dataset
        encode_function = partial(
            encode_flan_for_olmo,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            add_bos=args.add_bos,
        )
    elif (
        "prompt" in raw_datasets["train"].column_names
        and "completion" in raw_datasets["train"].column_names
    ):
        encode_function = partial(
            encode_with_prompt_completion_format,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            add_bos=args.add_bos,
        )
    elif "messages" in raw_datasets["train"].column_names:
        encode_function = partial(
            encode_with_messages_format,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            add_bos=args.add_bos,
        )
    else:
        raise ValueError(
            "You need to have either 'prompt'&'completion' or 'messages' in your column names."
        )

    with accelerator.main_process_first():
        if args.debug_small_scale:
            raw_datasets["train"] = raw_datasets["train"].select(range(10))
        lm_datasets = raw_datasets.map(
            encode_function,
            batched=False,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            remove_columns=[
                name
                for name in raw_datasets["train"].column_names
                if name not in ["input_ids", "labels", "attention_mask"]
            ],
            desc="Tokenizing and reformatting instruction data",
        )
        lm_datasets.set_format(type="pt")
        lm_datasets = lm_datasets.filter(
            lambda example: (example["labels"] != -100).any()
        )

    train_dataset = lm_datasets["train"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=DataCollatorForSeq2Seq(
            tokenizer=tokenizer, model=model, padding="longest"
        ),
        batch_size=args.per_device_train_batch_size,
        pin_memory=True, 
        num_workers=args.dataloader_num_workers,
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    if args.use_qlora:
        from bitsandbytes.optim import AdamW

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            optim_bits=8 if args.use_8bit_optimizer else 32,
            is_paged=True,
        )
    else:
        from accelerate.utils import DummyOptim, DummyScheduler

        optimizer_cls = (
            torch.optim.AdamW
            if accelerator.state.deepspeed_plugin is None
            or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
            else DummyOptim
        )
        optimizer = optimizer_cls(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # Create the learning rate scheduler.
    # Note: the current accelerator.step() calls the .step() of the real scheduler for the `num_processes` times. This is because they assume
    # the user initialize the scheduler with the entire training set. In the case of data parallel training, each process only
    # sees a subset (1/num_processes) of the training set. So each time the process needs to update the lr multiple times so that the total
    # number of updates in the end matches the num_training_steps here.
    # Here we need to set the num_training_steps to either using the entire training set (when epochs is specified) or we need to multiply the
    # num_training_steps by num_processes so that the total number of updates matches the num_training_steps.
    num_training_steps_for_scheduler = (
        args.max_train_steps
        if overrode_max_train_steps
        else args.max_train_steps * accelerator.num_processes
    )

    # Creates Dummy Scheduler if `scheduler` was specified in the config file else creates `args.lr_scheduler_type` Scheduler
    if (
        accelerator.state.deepspeed_plugin is None
        or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
    ):
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_training_steps=num_training_steps_for_scheduler,
            num_warmup_steps=int(num_training_steps_for_scheduler * args.warmup_ratio),
        )
    else:
        lr_scheduler = DummyScheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            total_num_steps=num_training_steps_for_scheduler,
            warmup_num_steps=int(num_training_steps_for_scheduler * args.warmup_ratio),
        )

    # Prepare everything with `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config[
            "lr_scheduler_type"
        ].value
        accelerator.init_trackers(
            "open_instruct",
            experiment_config,
            init_kwargs={
                "wandb": {"name": args.output_dir.split("/")[-1], "resume": "allow"}
            },
        )

    # Train!
    total_batch_size = (
        args.per_device_train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.per_device_train_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  max_seq_length = {args.max_seq_length}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(args.max_train_steps), disable=not accelerator.is_local_main_process
    )
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if (
            args.resume_from_checkpoint is not None or args.resume_from_checkpoint != ""
        ) and args.resume_from_checkpoint != "last":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            if args.resume_from_checkpoint == "last":
                output_dir = args.output_dir
            else:
                output_dir = os.getcwd()
            # Get the most recent checkpoint
            dirs = [
                os.path.join(output_dir, item)
                for item in os.listdir(output_dir)
                if os.path.isdir(os.path.join(output_dir, item))
            ]
            # Sorts folders by date modified, most recent checkpoint is the last
            # dirs.sort(key=os.path.getctime)
            # All folders are named step_{i} or epoch_{i}, sort according to the int i
            dirs.sort(key=lambda x: int(x.split("_")[-1]))
            try:
                path = dirs[-1]
            except IndexError:
                raise ValueError(
                    f"No checkpoint found in {output_dir}. Cannot resume training."
                )

            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        # model is peft model, need to load adapter weights
        if args.use_lora:
            accelerator_state_path = os.path.join(checkpoint_path, "accelerator_state")
            if os.path.exists(accelerator_state_path):
                # load accelerate with needed deepspeed config
                # accelerator.load_state(accelerator_state_path, optimizer=optimizer, lr_scheduler=lr_scheduler)
                accelerator.load_state(accelerator_state_path)
            # accelerator.load_state(path)
            # model.load_state_dict(torch.load(pytorch_model.bin))

            # model.load_adapter(checkpoint_path, adapter_name="adapter_name")
            # # set adapter
            # model.set_adapter("adapter_name")

        else:
            accelerator.load_state(path)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = (
                int(training_difference.replace("step_", ""))
                * args.gradient_accumulation_steps
            )
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        total_loss = 0
        if (
            args.resume_from_checkpoint
            and epoch == starting_epoch
            and resume_step is not None
        ):
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(
                train_dataloader, resume_step
            )
        else:
            active_dataloader = train_dataloader

        for step, batch in enumerate(active_dataloader):
            with accelerator.accumulate(model):
                # torch.cuda.empty_cache()
                # import gc

                # gc.collect()
                # print(torch.cuda.memory_summary())
                outputs = model(**batch, use_cache=False)

                if args.reduce_loss == "mean":
                    loss = outputs.loss
                else:
                    # reduce loss is sum
                    # this ensures that we weight all tokens in the dataset equally,
                    # rather than weighting each overall example equally when
                    # using high amounts of gradient accumulation.
                    # this can result in > 5 point improvements in AlpacaEval
                    # see https://github.com/huggingface/transformers/issues/24725 for
                    # more discussion and details.
                    logits = outputs.logits
                    labels = batch["labels"]
                    # Shift so that tokens < n predict n
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    # Flatten the tokens
                    loss_fct = torch.nn.CrossEntropyLoss(reduction="sum")
                    shift_logits = shift_logits.view(-1, embedding_size)
                    shift_labels = shift_labels.view(-1)
                    # Enable model parallelism
                    shift_labels = shift_labels.to(shift_logits.device)
                    loss = loss_fct(shift_logits, shift_labels)
                # We keep track of the loss at each logged step
                total_loss += loss.detach().float()
                accelerator.backward(loss)
                # clip gradient norm. don't do this with deepspeed
                if accelerator.sync_gradients and args.clip_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1
                if args.logging_steps and completed_steps % args.logging_steps == 0:
                    avg_loss = (
                        accelerator.gather(total_loss).mean().item()
                        / args.gradient_accumulation_steps
                        / args.logging_steps
                    )
                    logger.info(
                        f"  Step: {completed_steps}, LR: {lr_scheduler.get_last_lr()[0]}, Loss: {avg_loss}"
                    )
                    if args.with_tracking:
                        accelerator.log(
                            {
                                "learning_rate": lr_scheduler.get_last_lr()[0],
                                "train_loss": avg_loss,
                            },
                            step=completed_steps,
                        )
                    total_loss = 0

                if isinstance(checkpointing_steps, int):
                    if completed_steps % checkpointing_steps == 0:
                        output_dir = f"step_{completed_steps}"
                        if args.output_dir is not None:
                            output_dir = os.path.join(args.output_dir, output_dir)
                        save_with_accelerate(
                            accelerator,
                            model,
                            tokenizer,
                            output_dir,
                            args,
                            optimizer,
                            lr_scheduler,
                        )

                        # remove the previous checkpoint
                        if args.remove_previous_checkpoint:
                            previous_checkpoint = (
                                f"step_{completed_steps - checkpointing_steps}"
                            )
                            if args.output_dir is not None:
                                previous_checkpoint = os.path.join(
                                    args.output_dir, previous_checkpoint
                                )
                                # we remove the accelerator_state file only
                                accelerator_state_path = os.path.join(
                                    previous_checkpoint, "accelerator_state"
                                )
                                if os.path.exists(accelerator_state_path):
                                    # remove the dir recursively
                                    try:
                                        shutil.rmtree(accelerator_state_path)
                                    except OSError as e:
                                        logger.error(
                                            f"Error Deleteing: {e.filename} - {e.strerror}"
                                        )

                        # evaluate the model
                        accelerator.wait_for_everyone()
                        if (
                            args.evaluate_during_training
                            and accelerator.is_main_process
                        ):
                            weighted_acc = evaluate_model(
                                model,
                                tokenizer,
                                args,
                                accelerator,
                                step=completed_steps,
                            )
                            if args.with_tracking:
                                accelerator.log(
                                    {"MMLU Acc": weighted_acc}, step=completed_steps
                                )
                        accelerator.wait_for_everyone()

                if completed_steps >= args.max_train_steps:
                    break

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            save_with_accelerate(accelerator, model, tokenizer, output_dir, args)

    if args.with_tracking:
        accelerator.end_training()

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
        save_with_accelerate(
            accelerator,
            model,
            tokenizer,
            args.output_dir,
            args,
            optimizer,
            lr_scheduler,
        )


if __name__ == "__main__":
    main()
