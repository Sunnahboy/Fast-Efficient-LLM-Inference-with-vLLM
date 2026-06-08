import os
import time

# Enable the ultra-fast Rust transfer backend
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from huggingface_hub import snapshot_download

# Match the directory naming specified 
REPO_ID = "Qwen/Qwen3-0.6B"
LOCAL_DIR = "./models/Qwen3-0.6B"

print("=" * 60)
print(f" Starting high-speed model download via Rust hf-transfer...")
print(f"Target Repo:  {REPO_ID}")
print(f"Destination:  {LOCAL_DIR}")
print("=" * 60)

start_time = time.time()

try:
    snapshot_download(
        repo_id=REPO_ID,
        local_dir=LOCAL_DIR,
        local_dir_use_symlinks=False,
        ignore_patterns=["*.msgpack", "*.h5"] # Ignore unneeded non-pytorch weight files
    )
    
    elapsed_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(f" Success! Weights downloaded in {elapsed_time:.1f} seconds.")
    print(f"Your files are ready in: {os.path.abspath(LOCAL_DIR)}")
    print("=" * 60)

except Exception as e:
    print(f"\n Error during download: {e}")
