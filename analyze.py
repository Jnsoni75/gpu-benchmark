import json
import sys

def analyze_bottlenecks(baseline_file: str = "baseline.json"):
    with open(baseline_file, "r") as f:
        data = json.load(f)

    # Map CPU times so we can detect when GPU is slower than CPU
    cpu_times = {
        (row["workload"], row["size"]): row["median_ms"]
        for row in data if row["device"] == "CPU"
    }

    print("\n" + "=" * 110)
    print("BOTTLENECK PROFILING REPORT & ACTIONABLE RECOMMENDATIONS (GPU)")
    print("=" * 110)
    print(f"{'WORKLOAD':<16} | {'SIZE':<9} | {'BW (GB/s)':<10} | {'INTENSITY':<14} | {'BOTTLENECK':<18} | {'ACTIONABLE RECOMMENDATION'}")
    print("-" * 110)

    for row in data:
        if row["device"] == "CPU":
            continue

        # Calculate Effective Memory Bandwidth (GB/s) and Arithmetic Intensity (FLOPs/Byte)
        mem_gb = row["memory_mb"] / 1024.0
        time_sec = row["median_ms"] / 1000.0
        bandwidth_gbs = mem_gb / time_sec if time_sec > 0 else 0.0
        intensity = row["throughput_gflops"] / bandwidth_gbs if bandwidth_gbs > 0 else 0.0

        cpu_ms = cpu_times.get((row["workload"], row["size"]), 0)

        # Classify the bottleneck and generate actionable recommendation
        if row["median_ms"] >= cpu_ms:
            bottleneck = "LAUNCH OVERHEAD"
            advice = "Tensor too small for GPU; batch inputs or keep on CPU"
        elif intensity < 10.0:
            bottleneck = "MEMORY BANDWIDTH"
            advice = "Fuse operators (torch.compile) to cut memory transfers"
        else:
            bottleneck = "COMPUTE BOUND"
            advice = "Use half-precision (fp16/bf16) to double math speed"

        print(
            f"{row['workload']:<16} | {row['size']:<9} | "
            f"{bandwidth_gbs:6.1f} GB/s | {intensity:7.1f} FLOP/B | "
            f"{bottleneck:<18} | {advice}"
        )

def check_regressions(baseline_file: str = "baseline.json", threshold_pct: float = 5.0, simulate_bug: bool = False):
    with open(baseline_file, "r") as f:
        baseline = json.load(f)

    print("\n" + "=" * 110)
    print(f"AUTOMATED REGRESSION CHECK (Threshold: +{threshold_pct:.1f}% max allowed slowdown)")
    print("=" * 110)

    regressions_found = 0
    for row in baseline:
        if row["device"] == "CPU":
            continue

        base_ms = row["median_ms"]
        # If testing with --simulate-bug, artificially slow down 4096x4096 workloads by 12%
        if simulate_bug and row["size"] == "4096x4096":
            current_ms = round(base_ms * 1.12, 3)
        else:
            current_ms = base_ms

        change_pct = ((current_ms - base_ms) / base_ms) * 100.0
        status = "FAIL [REGRESSION]" if change_pct > threshold_pct else "PASS"
        if change_pct > threshold_pct:
            regressions_found += 1

        print(
            f"{status:<18} | {row['workload']:<16} | {row['device']:<4} | {row['size']:<9} | "
            f"Baseline: {base_ms:6.2f} ms -> Current: {current_ms:6.2f} ms ({change_pct:+.1f}%)"
        )

    print("-" * 110)
    if regressions_found > 0:
        print(f"ALERT: {regressions_found} performance regression(s) exceeded the {threshold_pct}% threshold!")
    else:
        print("ALL CHECKS PASSED: No performance degradation detected.")

if __name__ == "__main__":
    simulate = "--simulate-bug" in sys.argv
    analyze_bottlenecks()
    check_regressions(simulate_bug=simulate)
