import faulthandler
import os
import time
import torch
faulthandler.enable()
os.environ["DIFFSYNTH_DISK_MAP_BUFFER_SIZE"] = str(10**14)
from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig
from diffsynth.utils.data.audio_video import write_video_audio

BASE = "models/MiniMax-H3-NF4"

vram_config = {
    "offload_dtype": "disk",
    "offload_device": "disk",
    "onload_dtype": torch.bfloat16,
    "onload_device": "cpu",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}

print("loading pipeline ...")
t0 = time.time()
pipe = MiniMaxH3Pipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(path=f"{BASE}/minimax-h3-fl2va-nf4.safetensors", **vram_config),
        ModelConfig(path=f"{BASE}/minimax-h3-text-encoder-nf4.safetensors", **vram_config),
        ModelConfig(path=f"{BASE}/video_vae_nf4.safetensors", **vram_config),
        ModelConfig(path=f"{BASE}/audio_vae_nf4.safetensors", **vram_config),
    ],
    processor_config=ModelConfig(path=f"{BASE}/FL2VA/processor/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 4,
)
print(f"loaded in {time.time() - t0:.1f}s")

prompt = "A girl is very happy, she is speaking in english: I enjoy working with DiffSynth-Studio, it's a perfect framework."
print("generating (480x832, 124 frames, 50 steps) ...")
t0 = time.time()
video, audio = pipe(
    prompt=prompt,
    height=480, width=832, num_frames=124, num_inference_steps=50, seed=0,
)
print(f"generated in {time.time() - t0:.1f}s")
write_video_audio(
    video=video, audio=audio,
    output_path="h3_fl2va_test.mp4", fps=24, audio_sample_rate=32000,
)
print("saved h3_fl2va_test.mp4")
