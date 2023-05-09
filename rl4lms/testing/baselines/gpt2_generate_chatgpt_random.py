#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...)
on a text file or a dataset without using HuggingFace Trainer.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import argparse
import copy
import random

import pandas as pd
import logging
import os

import datasets
import torch

import transformers
from accelerate import Accelerator
from huggingface_hub import Repository
from tqdm import tqdm
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    set_seed,
)
# from transformers.file_utils import get_full_repo_name
from transformers.utils.versions import require_version

from rl4lms.envs.text_generation.evaluation_utils import get_batch
from rl4lms.envs.text_generation.registry import MetricRegistry
from rl4lms.envs.text_generation.training_utils import build_datapool
from rl4lms.testing.baselines.chatgpt import compute_and_dump_metrics, call_chatgpt

logger = logging.getLogger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def generate_text(model, tokenizer, batch, max_prompt_length, generation_kwargs):
    model.eval()
    prompt = """Continue this text into a positive movie review.\n{source}"""
    ans = []
    space_tensor = tokenizer(" ", return_tensors="pt")["input_ids"].to("cuda")
    for sample in batch:
        ind = random.randint(0, generation_kwargs["max_new_tokens"])
        input = prompt.format_map({"source": sample.prompt_or_input_text})
        if ind != 0:
            # generate using small model for few tokens
            ids = tokenizer(
                input,
                padding="max_length",
                max_length=max_prompt_length,
                return_tensors="pt",
                truncation=True,
            )["input_ids"].to("cuda")
            temp_gen_kwargs = copy.deepcopy(generation_kwargs)
            temp_gen_kwargs["max_new_tokens"] = ind
            temp_gen_kwargs["min_length"] = ind
            sample_outputs = model.generate(
                input_ids=ids,
                **temp_gen_kwargs
            )
            gen_text = tokenizer.decode(sample_outputs[0])

            # call big model for 10 tokens
            prompt_text = prompt.format_map({"source": gen_text})
            out = call_chatgpt(prompt_text, num_tokens=min(generation_kwargs["max_new_tokens"] - ind, 10))

            # generate using the small model
            ids = torch.cat([sample_outputs, space_tensor, tokenizer(out, return_tensors="pt")["input_ids"].to("cuda")],
                            dim=-1)
            temp_gen_kwargs = copy.deepcopy(generation_kwargs)
            temp_gen_kwargs["max_new_tokens"] = generation_kwargs["max_new_tokens"] - ind - 10
            temp_gen_kwargs["min_length"] = generation_kwargs["max_new_tokens"] - ind - 10
            sample_outputs = model.generate(
                input_ids=ids,
                **temp_gen_kwargs
            )
            gen_text = tokenizer.decode(sample_outputs[0][max_prompt_length:])
            ans.append(gen_text)
        else:
            # call big model for 10 tokens
            encodings = tokenizer(
                input,
                max_length=max_prompt_length,
                truncation=True,
            )
            input_truncs = tokenizer.decode(encodings["input_ids"], skip_special_tokens=True)
            prompt_text = prompt.format_map({"source": input_truncs})
            out = call_chatgpt(prompt_text, num_tokens=10)

            # generate using the small model
            input = prompt.format_map({"source": sample.prompt_or_input_text + " " + out})
            ids = tokenizer(
                input,
                padding="max_length",
                max_length=max_prompt_length + 10,
                return_tensors="pt",
                truncation=True,
            )["input_ids"].to("cuda")
            temp_gen_kwargs = copy.deepcopy(generation_kwargs)
            temp_gen_kwargs["max_new_tokens"] = generation_kwargs["max_new_tokens"] - 10
            temp_gen_kwargs["min_length"] = generation_kwargs["max_new_tokens"] - 10
            sample_outputs = model.generate(
                input_ids=ids,
                **temp_gen_kwargs
            )
            gen_text = tokenizer.decode(sample_outputs[0][max_prompt_length:])
            ans.append(gen_text)
    return ans


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--num_tries",
        type=int,
        default=1,
        help="Number of sequences to be generated for each Q&A",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--analyze_file", type=str, default=None, help="A csv or a json file containing the analysis data."
    )
    parser.add_argument(
        "--validation_split_percentage",
        default=5,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
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
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, required=True, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help="Optional input sequence length after tokenization. The training dataset will be truncated in block of this size for training. Default to the model max input length for single sentence inputs (take into account special tokens).",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", type=bool, default=False, help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--no_keep_linebreaks", action="store_true", help="Do not keep line breaks when using TXT files."
    )
    parser.add_argument(
        "--analyze_answers", action="store_true", help="Check answers in generated contexts"
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    args = parser.parse_args()

    # Sanity checks
    # if args.dataset_name is None and args.train_file is None and args.validation_file is None:
    #     raise ValueError("Need either a dataset name or a training/validation file.")
    # else:
    #     if args.train_file is not None:
    #         extension = args.train_file.split(".")[-1]
    #         assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, json or txt file."
    #     if args.validation_file is not None:
    #         extension = args.validation_file.split(".")[-1]
    #         assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, json or txt file."
    #
    # if args.push_to_hub:
    #     assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args


if __name__ == "__main__":
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    data_config = {
        "id": "imdb",
        "args": {"seed": 42, "prompt_prefix": "", "prompt_suffix": ""}
    }
    batch_size = 128
    max_prompt_length = 64
    metric_configs = [
        {"id": "learned_reward", "args": {"model_name": "lvwerra/distilbert-imdb", "label_ix": 1, "batch_size": 64}},
        {"id": "diversity", "args": {}},
    ]
    generation_kwargs = {
        "do_sample": True,
        "min_length": 48,
        "max_new_tokens": 48,
        "top_k": 50
    }

    metrics = [MetricRegistry.get(metric_config["id"], metric_config.get("args", {}))
               for metric_config in metric_configs]
    samples_by_split = build_datapool(data_config)
    samples = samples_by_split["test"]

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                # repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
                repo_name = ""
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    if args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config)

    model.to(device='cuda')
    model.resize_token_embeddings(len(tokenizer))

    all_generated_texts = []
    all_ref_texts = []
    all_prompt_texts = []
    for batch in tqdm(list(get_batch(samples, batch_size)), desc="Evaluating"):
        batch_generated_texts = generate_text(model, tokenizer, batch, max_prompt_length, generation_kwargs)
        batch_ref_texts = [sample.references for sample in batch]
        batch_prompt_texts = [sample.prompt_or_input_text for sample in batch]
        all_generated_texts.extend(batch_generated_texts)
        all_ref_texts.extend(batch_ref_texts)
        all_prompt_texts.extend(batch_prompt_texts)

    compute_and_dump_metrics(metrics, all_prompt_texts, all_generated_texts, all_ref_texts, args.output_dir)
    dic_df = pd.DataFrame.from_dict({"prompt": all_prompt_texts, "gen": all_generated_texts, "ref": all_ref_texts})
    dic_df.to_csv(os.path.join(args.output_dir, "out.csv"), index=False)
