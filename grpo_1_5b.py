"""Run GRPO training on the local GSM8K dataset."""

from pathlib import Path

from datasets import load_from_disk
from trl import GRPOConfig, GRPOTrainer
from trl.rewards import accuracy_reward

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_NAME = "openai/gsm8k"
DATASET_PATH = str(PROJECT_ROOT / "datasets" / "gsm8k" / "train")
MODEL_PATH = str(PROJECT_ROOT / "models" / "Qwen" / "Qwen2___5-1___5B-Instruct")


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
        output_dir="grpo-gsm8k",
        per_device_train_batch_size=4,
        num_generations=8,
        max_completion_length=512,
        logging_steps=10,
        num_train_epochs=1,
    )


def main():
    trainer = GRPOTrainer(
        model=MODEL_PATH,
        reward_funcs=accuracy_reward,
        train_dataset=load_train_dataset(),
        args=build_training_args(),
    )
    trainer.train()


if __name__ == "__main__":
    main()