"""Re-apply the local diffsynth patches required on Windows after `uv sync`.

Run after `uv sync`:
    uv run python apply_patches.py
"""
import os
from pathlib import Path

VENV = Path(".venv") / "Lib" / "site-packages"


def apply_computation_module_patch():
    path = VENV / "diffsynth" / "core" / "vram" / "layers.py"
    if not path.exists():
        raise SystemExit(f"not found: {path}")
    src = path.read_text(encoding="utf-8")
    old = (
        "        if self.disk_offload and device == \"disk\":\n"
        "            transient = self.quantize.backend.create_quantized_linear_shell(self.module, self.computation_dtype)\n"
        "            return self._load_from_disk(self.computation_device, target=transient)\n"
    )
    new = (
        "        if self.disk_offload:\n"
        "            transient = self.quantize.backend.create_quantized_linear_shell(self.module, self.computation_dtype)\n"
        "            return self._load_from_disk(self.computation_device, target=transient)\n"
    )
    if new in src:
        print("[already patched] layers.py computation_module")
        return
    if old not in src:
        raise SystemExit("layers.py: expected snippet not found (diffsynth version changed?)")
    path.write_text(src.replace(old, new), encoding="utf-8")
    print("[patched] layers.py computation_module -> disk-offload modules load from disk, no deepcopy")


if __name__ == "__main__":
    apply_computation_module_patch()
    print("done")
