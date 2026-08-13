# MiniMax-H3 on RTX 3080 Laptop (16GB VRAM)

Run MiniMax-H3 FL2VA (text/audio -> video + synchronized audio) locally with
PyTorch CUDA, DiffSynth-Studio, and NF4 quantization.

Verified on: Windows 11, NVIDIA GeForce RTX 3080 Laptop GPU 16GB, 31GB RAM,
driver 595.97 (CUDA 13.2).

## Requirements

- `uv` (>= 0.11)
- NVIDIA GPU with >= 8GB VRAM (16GB used here)
- ~50GB free disk (~33GB models + venv)
- CUDA-capable NVIDIA driver

## Setup

```powershell
# 1. Create venv (Python 3.12, in PyTorch's supported 3.10-3.14 range)
uv venv .venv --python 3.12

# 2. Install CUDA PyTorch (cu132 matches driver 13.2; falls back to cu130)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132

# 3. Install DiffSynth + bitsandbytes
uv pip install "diffsynth[all]" bitsandbytes huggingface_hub
```

## Model download (~33GB)

```powershell
uv run python download_model.py
```

Downloads (to `models/MiniMax-H3-NF4/`):

| File | Size |
|---|---|
| `minimax-h3-fl2va-nf4.safetensors` | 16.4GB |
| `minimax-h3-text-encoder-nf4.safetensors` | 14.6GB |
| `video_vae_nf4.safetensors` | 1.5GB |
| `audio_vae_nf4.safetensors` | 271MB |
| `FL2VA/processor/` | tokenizer config |

Plus the Turbo LoRA `models/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors`.

## Run

Base model (50 steps, slow, ~40-90 min per clip on this GPU):

```powershell
uv run python run_h3.py
```

Turbo LoRA (8 steps, ~4 min per clip, recommended):

```powershell
uv run python run_h3_turbo.py
```

Output: `h3_fl2va_turbo_test.mp4` (640x480, 24fps, H264 + 32kHz stereo AAC).

### Turbo settings

`flow_shift=12`, `audio_flow_shift=3`, `num_inference_steps=8`. DiffSynth's
MiniMax-H3 scheduler uses the same NFE grid (`q=(N-i)/N`) as the Turbo
distillation, so no custom sampler is needed.

## Patches applied to diffsynth (Windows workarounds)

1. **Disk-offload mmap crash** — `SafetensorsDiskMap.flush_files()` re-opens
   the `.safetensors` when >1e9 params are read, invalidating lazy mmap views
   (access violation in `torch_cpu.dll`). Fix: set
   `DIFFSYNTH_DISK_MAP_BUFFER_SIZE=10**14` to disable the flush (set in the
   run scripts).

2. **`copy.deepcopy` crash in VRAM manager** — `layers.py:computation_module()`
   deep-copies quantized bnb layers when VRAM is tight, crashing in
   `torch.storage.clone`. Patch in
   `.venv/Lib/site-packages/diffsynth/core/vram/layers.py`: for disk-offloaded
   modules, always load from disk to the compute device instead of deep-copy:

   ```python
   if self.disk_offload:
       transient = self.quantize.backend.create_quantized_linear_shell(self.module, self.computation_dtype)
       return self._load_from_disk(self.computation_device, target=transient)
   ```

3. **Turbo LoRA key mismatch** — the diffusers-format
   `minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors` uses separate
   `to_q/to_k/to_v` projections that don't match DiffSynth's fused
   `qkv_proj`. Use the **ComfyUI** variant
   `minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors` (fused keys,
   `diffusion_model.` prefix auto-stripped by `GeneralLoRALoader`).

## Notes

- 50-step base generation is impractical on 16GB VRAM (disk reloading of the
  33B NF4 DiT per step). Use the Turbo LoRA.
- cu132 index currently lacks Windows `torchaudio` wheels; it is not needed
  here.
- Ref2VA (image/video/audio reference) is supported by the NF4 repo but not
  downloaded in this setup (adds another 16.4GB).
