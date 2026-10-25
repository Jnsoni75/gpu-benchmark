import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#CCCCCC'
plt.rcParams['axes.linewidth'] = 0.8

fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), dpi=200)
fig.patch.set_facecolor('#FFFFFF')

c_cpu, c_cuda, c_mps = '#64748B', '#10B981', '#3B82F6'

# Panel 1: Throughput Comparison (4096x4096)
ax1 = axes[0]
workloads = ['MatMul\n(Compute-Bound)', 'MLP NeuralNet\n(Mixed AI Model)', 'Elementwise Add\n(Memory-Bound)']
x = np.arange(len(workloads))
width = 0.25

cpu_gflops = [2888.2, 2276.2, 10.9]
cuda_gflops = [3897.2, 3259.7, 20.4]
mps_gflops = [10183.3, 8302.5, 22.4]

r1 = ax1.bar(x - width, cpu_gflops, width, label='CPU Baseline', color=c_cpu, zorder=3)
r2 = ax1.bar(x, cuda_gflops, width, label='NVIDIA Tesla T4 (CUDA)', color=c_cuda, zorder=3)
r3 = ax1.bar(x + width, mps_gflops, width, label='Apple Silicon (MPS)', color=c_mps, zorder=3)

ax1.set_title('Peak Compute Throughput (4096×4096)', fontsize=13, fontweight='bold', pad=14)
ax1.set_ylabel('Throughput (GFLOPS — Higher is Better)', fontsize=10.5, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(workloads, fontsize=9.5, fontweight='bold')
ax1.set_ylim(0, 11800)
ax1.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
ax1.legend(fontsize=9, loc='upper right')

for rect in r1 + r2 + r3:
    h = rect.get_height()
    ax1.annotate(f"{h:,.0f}" if h >= 100 else f"{h:.1f}",
                 xy=(rect.get_x() + rect.get_width() / 2, h),
                 xytext=(0, 4), textcoords="offset points",
                 ha='center', va='bottom', fontsize=8, fontweight='bold')

# Panel 2: MatMul Latency Scaling (Log Scale)
ax2 = axes[1]
sizes = ['256×256\n(Launch Overhead)', '1024×1024\n(Crossover)', '4096×4096\n(Parallel Scaling)']
x2 = np.arange(len(sizes))

ax2.plot(x2, [0.47, 17.38, 1153.87], 'o--', lw=2, color='#94A3B8', label='Colab CPU (Xeon)')
ax2.plot(x2, [0.02, 0.84, 47.59], 's--', lw=2, color=c_cpu, label='MacBook CPU (Apple)')
ax2.plot(x2, [0.08, 0.83, 35.27], 'o-', lw=2.5, color=c_cuda, label='NVIDIA Tesla T4 (CUDA)')
ax2.plot(x2, [0.18, 0.40, 13.50], 's-', lw=2.5, color=c_mps, label='Apple Silicon GPU (MPS)')

ax2.set_yscale('log')
ax2.set_title('MatMul Latency Scaling across Tensor Sizes', fontsize=13, fontweight='bold', pad=14)
ax2.set_ylabel('Median Latency in ms (Log Scale — Lower is Better)', fontsize=10.5, fontweight='bold')
ax2.set_xticks(x2)
ax2.set_xticklabels(sizes, fontsize=9.5, fontweight='bold')
ax2.grid(True, which='both', linestyle='--', alpha=0.4, zorder=0)
ax2.legend(fontsize=8.5, loc='upper left')

# Panel 3: Effective Memory Bandwidth (GB/s)
ax3 = axes[2]
workloads_bw = ['MatMul\n(Compute-Bound)', 'MLP NeuralNet\n(Compute-Bound)', 'Elementwise Add\n(Bandwidth Ceiling)']
x3 = np.arange(len(workloads_bw))
width3 = 0.32

rb1 = ax3.bar(x3 - width3/2, [5.7, 26.3, 244.6], width3, label='NVIDIA Tesla T4 (CUDA)', color=c_cuda, zorder=3)
rb2 = ax3.bar(x3 + width3/2, [13.9, 66.1, 250.0], width3, label='Apple Silicon (MPS)', color=c_mps, zorder=3)
ax3.axhline(250, color='#EF4444', linestyle=':', lw=1.5, label='Hardware Bandwidth Ceiling (~250 GB/s)')

ax3.set_title('Effective Memory Bandwidth (4096×4096)', fontsize=13, fontweight='bold', pad=14)
ax3.set_ylabel('Memory Bandwidth (GB/s — Higher is Better)', fontsize=10.5, fontweight='bold')
ax3.set_xticks(x3)
ax3.set_xticklabels(workloads_bw, fontsize=9.5, fontweight='bold')
ax3.set_ylim(0, 295)
ax3.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
ax3.legend(fontsize=8.5, loc='upper left')

for rect in rb1 + rb2:
    h = rect.get_height()
    ax3.annotate(f"{h:.1f} GB/s",
                 xy=(rect.get_x() + rect.get_width() / 2, h),
                 xytext=(0, 4), textcoords="offset points",
                 ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.suptitle('GPU Benchmarking & Bottleneck Analysis: NVIDIA CUDA (Tesla T4) vs. Apple Metal (MPS) vs. CPU',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('gpu_benchmark_comparison.png', dpi=200, bbox_inches='tight')
print("Saved gpu_benchmark_comparison.png!")
