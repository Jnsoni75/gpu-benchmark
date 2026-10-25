import time
import json
import statistics
import torch
import torch.nn as nn

def synchronize(device_name: str):
    if device_name == "mps":
        torch.mps.synchronize()
    elif device_name == "cuda":
        torch.cuda.synchronize()

def get_memory_mb(device_name: str, tensors: list) -> float:
    if device_name == "cuda":
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    elif device_name == "mps":
        return torch.mps.current_allocated_memory() / (1024 ** 2)
    else:
        total_bytes = sum(t.element_size() * t.nelement() for t in tensors)
        return total_bytes / (1024 ** 2)

def run_workload(workload_name: str, device_name: str, size: int, warmup: int = 10, runs: int = 40):
    device = torch.device(device_name)
    if device_name == "cuda":
        torch.cuda.reset_peak_memory_stats()

    if workload_name == "MatMul":
        a = torch.randn((size, size), device=device)
        b = torch.randn((size, size), device=device)
        fn = lambda: torch.matmul(a, b)
        flops = 2 * (size ** 3)
        tracked_items = [a, b]
    elif workload_name == "Elementwise_Add":
        a = torch.randn((size, size), device=device)
        b = torch.randn((size, size), device=device)
        fn = lambda: a + b
        flops = size ** 2
        tracked_items = [a, b]
    elif workload_name == "MLP_NeuralNet":
        model = nn.Sequential(
            nn.Linear(size, size),
            nn.ReLU(),
            nn.Linear(size, size)
        ).to(device)
        x = torch.randn((256, size), device=device)
        fn = lambda: model(x)
        flops = 2 * 256 * (size ** 2) * 2
        tracked_items = [x] + list(model.parameters())

    # 1. Warm-up runs to bypass allocator and JIT cold-start overhead
    for _ in range(warmup):
        _ = fn()
    synchronize(device_name)

    # 2. Peak memory measurement
    mem_mb = get_memory_mb(device_name, tracked_items)

    # 3. Timed execution with NVTX annotations for Nsight Systems / Compute
    times_ms = []
    if device_name == "cuda":
        torch.cuda.nvtx.range_push(f"{workload_name}_{size}")
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(runs)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(runs)]
        for i in range(runs):
            start_events[i].record()
            _ = fn()
            end_events[i].record()
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
        times_ms = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    else:
        for _ in range(runs):
            start = time.perf_counter()
            _ = fn()
            synchronize(device_name)
            end = time.perf_counter()
            times_ms.append((end - start) * 1000.0)

    median_ms = statistics.median(times_ms)
    avg_ms = statistics.mean(times_ms)
    std_ms = statistics.stdev(times_ms)
    p95_ms = sorted(times_ms)[int(0.95 * len(times_ms))]
    gflops = (flops / (median_ms / 1000.0)) / 1e9

    result = {
        "workload": workload_name,
        "device": device_name.upper(),
        "size": f"{size}x{size}",
        "median_ms": round(median_ms, 3),
        "avg_ms": round(avg_ms, 3),
        "std_ms": round(std_ms, 3),
        "p95_ms": round(p95_ms, 3),
        "throughput_gflops": round(gflops, 2),
        "memory_mb": round(mem_mb, 2)
    }

    print(
        f"{workload_name:<16} | {device_name.upper():<4} | {size:<4}x{size:<4} | "
        f"Med: {median_ms:7.2f} ms | Speed: {gflops:8.1f} GFLOPS | Mem: {mem_mb:6.1f} MB"
    )
    return result

if __name__ == "__main__":
    gpu = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else None)
    devices = ["cpu", gpu] if gpu else ["cpu"]
    workloads = ["MatMul", "Elementwise_Add", "MLP_NeuralNet"]
    sizes = [256, 1024, 4096]

    all_results = []
    print("=" * 92)
    print(f"{'WORKLOAD':<16} | {'DEV':<4} | {'SIZE':<9} | {'MEDIAN TIME':<14} | {'THROUGHPUT':<19} | {'MEMORY'}")
    print("=" * 92)

    for w in workloads:
        for s in sizes:
            for d in devices:
                res = run_workload(w, d, s)
                all_results.append(res)
        print("-" * 92)

    with open("baseline.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved benchmark results to baseline.json!")
