import time
import torch

print("torch:", torch.__version__, "| cuda:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0))
print()

def bench(size, n=5):
    a = torch.randn(size, size, device="cuda")
    b = torch.randn(size, size, device="cuda")
    for _ in range(2):
        torch.matmul(a, b)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        torch.matmul(a, b)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / n
    flops = 2.0 * size**3
    print(f"{size:>6}x{size:<6} avg {dt*1000:8.2f} ms   {flops/dt/1e12:7.2f} TFLOPS")

for size in (1024, 2048, 4096, 8192):
    bench(size)
