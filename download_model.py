import os
from modelscope import snapshot_download

model_dir = snapshot_download(
    "Qwen/Qwen2.5-1.5B-Instruct",
    cache_dir=os.path.join(os.path.dirname(__file__), "models")
)
print(f"模型下载完成: {model_dir}")