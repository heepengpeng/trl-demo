"""Download the raw GSM8K train split into the project directory."""

from pathlib import Path

from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_NAME = "openai/gsm8k"
DATASET_CONFIG = "main"
DATASET_PATH = PROJECT_ROOT / "datasets" / "gsm8k" / "train"


def main():
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split="train")
    dataset.save_to_disk(str(DATASET_PATH))


if __name__ == "__main__":
    main()
