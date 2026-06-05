"""Run GRPO training on the local GSM8K dataset."""

import re
from pathlib import Path

from datasets import load_from_disk
from trl import GRPOConfig, GRPOTrainer
from trl.rewards import accuracy_reward as trl_accuracy_reward

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_NAME = "openai/gsm8k"
DATASET_PATH = str(PROJECT_ROOT / "datasets" / "gsm8k" / "train")
MODEL_PATH = str(PROJECT_ROOT / "models" / "Qwen" / "Qwen2___5-0___5B-Instruct")


def format_gsm8k_example(example):
    """Format a GSM8K row for TRL GRPO prompts and accuracy rewards."""
    return {
        "prompt": [{"role": "user", "content": example["question"]}],
        "solution": example["answer"],
    }


def load_train_dataset():
    dataset = load_from_disk(DATASET_PATH)
    return dataset.map(format_gsm8k_example)


def build_training_args():
    return GRPOConfig(
        output_dir="grpo-checkpoints",
        max_steps=1000,
        per_device_train_batch_size=8,
        num_generations=8,
        max_completion_length=256,
        bf16=True,
    )


def extract_last_number(text):
    matches = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text)
    if not matches:
        return None
    return matches[-1].replace(",", "")


def get_completion_content(completion):
    if isinstance(completion, str):
        return completion
    return completion[0]["content"]


def gsm8k_numeric_reward(completions, solution, **kwargs):
    rewards = []
    for completion, expected_solution in zip(completions, solution, strict=True):
        predicted = extract_last_number(get_completion_content(completion))
        expected = extract_last_number(expected_solution.split("####")[-1])
        rewards.append(1.0 if predicted is not None and predicted == expected else 0.0)
    return rewards


def accuracy_reward(completions, solution, **kwargs):
    if completions and isinstance(completions[0], str):
        completions = [
            [{"role": "assistant", "content": completion}]
            for completion in completions
        ]

    return trl_accuracy_reward(completions=completions, solution=solution, **kwargs)


def main():
    trainer = GRPOTrainer(
        model=MODEL_PATH,
        args=build_training_args(),
        reward_funcs=[gsm8k_numeric_reward, accuracy_reward],
        train_dataset=load_train_dataset(),
    )
    trainer.train()


if __name__ == "__main__":
    main()