#!/usr/bin/env python3
"""Benchmark: transformer entrenado vs Python vs C para aritmética modular."""

import json
import time
import subprocess
import os
import sys
import torch

sys.path.insert(0, '/home/kota/investigacion/fase2/grokking')
from modelo import GrokTransformer

P = 97
results = {}

# A) Transformer inference speed
print("=== Transformer Inference ===")
model = GrokTransformer(p=P, dim=128, n_layers=2, n_heads=1, ff_dim=512)

# Try to load trained model
model_path = "/home/kota/investigacion/fase2/grokking/modelo_mod_add.pt"
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, weights_only=True))
    print(f"Loaded trained model from {model_path}")
else:
    print("No trained model found, using random weights (speed test only)")

model.eval()

# Warmup
a = torch.tensor([5])
b = torch.tensor([3])
with torch.no_grad():
    for _ in range(100):
        model(a, b)

# Single inference benchmark
times = []
for _ in range(1000):
    start = time.perf_counter()
    with torch.no_grad():
        model(a, b)
    end = time.perf_counter()
    times.append((end - start) * 1e6)

median_us = sorted(times)[500]
print(f"Single inference: {median_us:.0f} μs")

# Batch inference
batch_a = torch.randint(0, P, (1000,))
batch_b = torch.randint(0, P, (1000,))
with torch.no_grad():
    for _ in range(10):  # warmup
        model(batch_a, batch_b)

batch_times = []
for _ in range(100):
    start = time.perf_counter()
    with torch.no_grad():
        model(batch_a, batch_b)
    end = time.perf_counter()
    batch_times.append((end - start))

batch_median = sorted(batch_times)[50]
ops_per_sec = 1000 / batch_median

print(f"Batch 1000: {batch_median*1e6:.0f} μs ({ops_per_sec:.0f} ops/sec)")

results["transformer"] = {
    "single_inference_us": round(median_us, 1),
    "batch_1000_us": round(batch_median * 1e6, 1),
    "ops_per_sec": round(ops_per_sec),
    "params": model.count_params()
}

# B) Python native
print("\n=== Python Native ===")
N = 10_000_000
start = time.perf_counter()
for i in range(N):
    _ = (i + 42) % P
end = time.perf_counter()
python_time = end - start
python_ops = N / python_time
python_per_op_us = (python_time / N) * 1e6
print(f"(a+b)%p: {python_per_op_us:.4f} μs/op ({python_ops:.0f} ops/sec)")

results["python_native"] = {
    "per_op_us": round(python_per_op_us, 4),
    "ops_per_sec": round(python_ops),
    "iterations": N
}

# C) C compiled
print("\n=== C Compiled ===")
c_code = """
#include <stdio.h>
#include <time.h>
#define P 97
int main() {
    long long N = 100000000;
    struct timespec start, end;
    volatile int result = 0;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (long long i = 0; i < N; i++) {
        result = (int)((i + 42) % P);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("Iterations: %lld\\n", N);
    printf("Time: %.3f seconds\\n", elapsed);
    printf("Speed: %.0f ops/sec\\n", N / elapsed);
    printf("Per op: %.4f ns\\n", (elapsed / N) * 1e9);
    return result & 0;
}
"""
c_path = "/home/kota/investigacion/fase2/grokking/modadd_bench.c"
bin_path = "/home/kota/investigacion/fase2/grokking/modadd_bench"
with open(c_path, "w") as f:
    f.write(c_code)

comp = subprocess.run(f"gcc -O3 -march=native -o {bin_path} {c_path}", shell=True, capture_output=True, text=True)
if comp.returncode == 0:
    run = subprocess.run(bin_path, capture_output=True, text=True, timeout=30)
    print(run.stdout)
    lines = run.stdout.strip().split('\n')
    c_ops = float([l for l in lines if "Speed:" in l][0].split(":")[1].strip().split()[0])
    c_per_op_ns = float([l for l in lines if "Per op:" in l][0].split(":")[1].strip().split()[0])
    results["c_native"] = {
        "per_op_us": round(c_per_op_ns / 1000, 4),
        "ops_per_sec": round(c_ops),
        "iterations": 100000000
    }
else:
    print(f"Compile error: {comp.stderr}")
    results["c_native"] = {"error": comp.stderr}

# D) Ratios
print("\n=== Comparación ===")
if "c_native" in results and "error" not in results["c_native"]:
    ratio_vs_c = results["transformer"]["single_inference_us"] / results["c_native"]["per_op_us"]
    print(f"Transformer es {ratio_vs_c:.0f}x más lento que C")
    results["ratios"] = {
        "transformer_vs_c": round(ratio_vs_c, 1),
        "transformer_vs_python": round(results["transformer"]["single_inference_us"] / results["python_native"]["per_op_us"], 1),
        "python_vs_c": round(results["python_native"]["per_op_us"] / results["c_native"]["per_op_us"], 1)
    }

with open("/home/kota/investigacion/fase2/grokking/benchmark_resultados.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nGuardado en benchmark_resultados.json")
