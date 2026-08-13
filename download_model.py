import os
from huggingface_hub import hf_hub_download, snapshot_download

os.makedirs("models/MiniMax-H3-NF4", exist_ok=True)

files = [
    "minimax-h3-fl2va-nf4.safetensors",
    "minimax-h3-text-encoder-nf4.safetensors",
    "video_vae_nf4.safetensors",
    "audio_vae_nf4.safetensors",
]
for f in files:
    dst = os.path.join("models/MiniMax-H3-NF4", f)
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        print(f"skip {f}")
        continue
    print(f"downloading {f} ...")
    hf_hub_download(
        "DiffSynth-Studio/MiniMax-H3-NF4", f, local_dir="models/MiniMax-H3-NF4",
    )
    print(f"done {f}")

print("downloading processor ...")
snapshot_download(
    "MiniMax/MiniMax-H3",
    allow_patterns="FL2VA/processor/*",
    local_dir="models/MiniMax-H3-NF4",
)
print("ALL DONE")
