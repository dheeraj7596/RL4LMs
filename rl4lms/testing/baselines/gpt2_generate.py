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
import pandas as pd
import logging
import os

import datasets

import transformers
import yaml
from accelerate import Accelerator
from tqdm import tqdm
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    set_seed,
)
# from transformers.file_utils import get_full_repo_name
from transformers.utils.versions import require_version

from rl4lms.envs.text_generation.evaluation_utils import get_batch
from rl4lms.envs.text_generation.registry import MetricRegistry
from rl4lms.envs.text_generation.training_utils import build_datapool, build_tokenizer
from rl4lms.testing.baselines.chatgpt import compute_and_dump_metrics

logger = logging.getLogger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def generate_text(model, tokenizer, batch, max_prompt_length, generation_kwargs):
    model.eval()
    EOS_TOKEN = '<EOS>'
    prompt = """{source}"""
    inputs = [prompt.format_map({"source": sample.prompt_or_input_text}) for sample in batch]
    inputs_tok = tokenizer(
        inputs,
        padding="max_length",
        max_length=max_prompt_length,
        return_tensors="pt",
        truncation=True,
    )
    ids = inputs_tok["input_ids"].to("cuda")
    attn_mask = inputs_tok["attention_mask"].to("cuda")
    sample_outputs = model.generate(
        input_ids=ids,
        attention_mask=attn_mask,
        **generation_kwargs
    )
    ans = []
    for t in sample_outputs:
        gen_text = tokenizer.decode(t[max_prompt_length:], skip_special_tokens=True)
        gen_text = gen_text.split(EOS_TOKEN)[0].strip()
        ans.append(gen_text)
    return ans


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
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
        "--config_path", type=str, help="path to the config file"
    )
    parser.add_argument("--output_dir", type=str, default=None, required=True, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    args = parser.parse_args()

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

    with open(args.config_path, "r") as fp:
        config = yaml.safe_load(fp)

    data_config = config["datapool"]
    batch_size = config.get("batch_size", 128)

    metric_configs = config["metrics"]
    metrics = [MetricRegistry.get(metric_config["id"], metric_config.get("args", {}))
               for metric_config in metric_configs]

    samples_by_split = build_datapool(data_config)
    samples = samples_by_split[config.get("split", "test")]

    generation_kwargs = config["generation_kwargs"]

    tokenizer_config = config["tokenizer"]
    tokenizer = build_tokenizer(tokenizer_config)
    max_prompt_length = config.get("max_prompt_length", tokenizer.model_max_length)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
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
