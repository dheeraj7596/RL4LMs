import json
from argparse import ArgumentParser

import openai
import os

import pandas as pd
import yaml
from ratelimiter import RateLimiter
from rl4lms.envs.text_generation.registry import MetricRegistry
from rl4lms.envs.text_generation.training_utils import build_datapool
from rl4lms.envs.text_generation.evaluation_utils import get_batch
from transformers import AutoTokenizer
from tqdm import tqdm


@RateLimiter(max_calls=600, period=60)
def call_chatgpt(input, num_tokens):
    try:
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                                  messages=[
                                                      {
                                                          "role": "user",
                                                          "content": input
                                                      }
                                                  ],
                                                  max_tokens=num_tokens)
        cont_gpt3 = completion.choices[0].message.content
    except Exception as e:
        print("OPENAI request failed", e)
        cont_gpt3 = ""
    return cont_gpt3


def generate_text(prompt, tokenizer, samples, max_prompt_length, num_tokens):
    inputs = [sample.prompt_or_input_text for sample in samples]
    encodings = tokenizer(
        inputs,
        max_length=max_prompt_length,
        truncation=True,
    )
    input_truncs = tokenizer.batch_decode(encodings["input_ids"], skip_special_tokens=True)

    generated_texts = []
    for i in range(len(samples)):
        prompt_text = prompt + input_truncs[i]
        out = call_chatgpt(prompt_text, num_tokens)
        generated_texts.append(out)

    return generated_texts


def compute_and_dump_metrics(metrics, all_prompt_texts, all_generated_texts, all_ref_texts, out_path):
    n_samples = len(all_generated_texts)
    corpus_level_metrics = {}
    sample_scores_by_metric = {}
    if metrics is not None:
        for metric in metrics:
            metric_dict = metric.compute(
                all_prompt_texts,
                all_generated_texts,
                all_ref_texts,
            )

            for metric_key, (sample_scores, corpus_score) in metric_dict.items():
                if sample_scores is None:
                    sample_scores = ["n/a"] * n_samples
                corpus_level_metrics[metric_key] = corpus_score
                sample_scores_by_metric[metric_key] = sample_scores
    with open(os.path.join(out_path, "corpus_metrics.json"), "w") as f:
        json.dump(corpus_level_metrics, f)
    with open(os.path.join(out_path, "sample_metrics.json"), "w") as f:
        json.dump(sample_scores_by_metric, f)


if __name__ == "__main__":
    parser = ArgumentParser(description="Run ChatGPT for tasks")
    parser.add_argument("--config_path", type=str, help="path to the config file")
    parser.add_argument(
        "--out_path", type=str, help="Directory to save results"
    )
    args = parser.parse_args()

    with open(args.config_path, "r") as fp:
        config = yaml.safe_load(fp)

    os.makedirs(args.out_path, exist_ok=True)
    data_config = config["datapool"]
    metric_configs = config["metrics"]

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    max_prompt_length = config.get("max_prompt_length", tokenizer.model_max_length)
    num_tokens = config["num_tokens"]
    metrics = [MetricRegistry.get(metric_config["id"], metric_config.get("args", {}))
               for metric_config in metric_configs]
    samples_by_split = build_datapool(data_config)
    samples = samples_by_split[config.get("split", "test")]

    all_generated_texts = []
    all_ref_texts = []
    all_prompt_texts = []
    for batch in tqdm(list(get_batch(samples, 1)), desc="Evaluating"):
        batch_generated_texts = generate_text(
            config.get("prefix", ""),
            tokenizer,
            batch,
            max_prompt_length,
            num_tokens
        )
        batch_ref_texts = [sample.references for sample in batch]
        batch_prompt_texts = [sample.prompt_or_input_text for sample in batch]
        all_generated_texts.extend(batch_generated_texts)
        all_ref_texts.extend(batch_ref_texts)
        all_prompt_texts.extend(batch_prompt_texts)

    compute_and_dump_metrics(metrics, all_prompt_texts, all_generated_texts, all_ref_texts, args.out_path)
    dic_df = pd.DataFrame.from_dict({"prompt": all_prompt_texts, "gen": all_generated_texts, "ref": all_ref_texts})
    dic_df.to_csv(os.path.join(args.out_path, "out.csv"), index=False)
