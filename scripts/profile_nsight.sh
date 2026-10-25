#!/usr/bin/env bash
set -e

echo "=== 1. Running NVIDIA Nsight Systems (Timeline & NVTX Trace) ==="
nsys profile \
  --trace=cuda,nvtx,osrt \
  --stats=true \
  --force-overwrite=true \
  -o nsight_systems_report \
  python benchmark.py

echo "=== 2. Running NVIDIA Nsight Compute (Kernel Roofline & Occupancy) ==="
ncu \
  --set full \
  --target-processes all \
  --force-overwrite \
  -o nsight_compute_report \
  python benchmark.py
