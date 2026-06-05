import logging
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_dataset
from modelscope import snapshot_download
from trl import SFTConfig, SFTTrainer

logger = logging.getLogger(__name__)


def plot_training_loss(trainer, output_path):
    steps = []
    losses = []
    for entry in trainer.state.log_history:
        if "loss" not in entry:
            continue
        steps.append(entry.get("step", len(steps)))
        losses.append(entry["loss"])

    if not losses:
        logger.warning("训练日志中没有 loss，跳过绘图")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(steps, losses, linewidth=1.5)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info("Loss 曲线已保存到 %s", output_path)


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

model_path = os.environ.get("MODEL_PATH") or snapshot_download("Qwen/Qwen2.5-0.5B")

train_dataset = load_dataset("trl-lib/Capybara", split="train")
max_samples = int(os.environ.get("MAX_SAMPLES", "2000"))
if max_samples > 0:
    train_dataset = train_dataset.select(range(min(max_samples, len(train_dataset))))

output_dir = "checkpoints"
sft_kwargs = {
    "output_dir": output_dir,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "max_length": 512,
    "loss_type": "chunked_nll",
    "packing": True,
    "num_train_epochs": int(os.environ.get("NUM_TRAIN_EPOCHS", "1")),
    "logging_steps": 10,
    "dataloader_num_workers": 4,
}
if max_steps := os.environ.get("MAX_STEPS"):
    sft_kwargs["max_steps"] = int(max_steps)

trainer = SFTTrainer(
    model=model_path,
    args=SFTConfig(**sft_kwargs),
    train_dataset=train_dataset,
)
trainer.train()
plot_training_loss(trainer, os.path.join(output_dir, "loss.png"))
