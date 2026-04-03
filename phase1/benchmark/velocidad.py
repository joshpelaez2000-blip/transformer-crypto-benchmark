#!/usr/bin/env python3
"""Benchmark: GPT-2 forward pass vs SHA-256 vs SHA-256 en C."""

import json
import time
import hashlib
import subprocess
import os
import torch
import numpy as np

results = {}

# ===== A) GPT-2 Forward Pass =====
print("=== GPT-2 Forward Pass ===")
from transformers import GPT2Model, GPT2Tokenizer

model = GPT2Model.from_pretrained('gpt2')
model.eval()
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Single token input
input_ids = torch.tensor([[50256]])  # <|endoftext|> token

# Warmup
with torch.no_grad():
    for _ in range(5):
        _ = model(input_ids)

# Benchmark
times = []
for _ in range(100):
    start = time.perf_counter()
    with torch.no_grad():
        _ = model(input_ids)
    end = time.perf_counter()
    times.append((end - start) * 1e6)  # microseconds

gpt2_us = np.median(times)
gpt2_min = np.min(times)
gpt2_max = np.max(times)

# Load FLOPs from paso 2
with open("/home/kota/investigacion/fase1/transformer/mapa_operaciones.json") as f:
    transformer_data = json.load(f)
gpt2_flops = transformer_data["total_ops_por_token"]["total_flops"]

print(f"Median: {gpt2_us:.0f} us")
print(f"Min: {gpt2_min:.0f} us, Max: {gpt2_max:.0f} us")
print(f"FLOPs: {gpt2_flops:,}")
print(f"GFLOPS: {gpt2_flops / (gpt2_us * 1e-6) / 1e9:.2f}")

results["gpt2_forward_pass"] = {
    "median_us": round(gpt2_us, 1),
    "min_us": round(gpt2_min, 1),
    "max_us": round(gpt2_max, 1),
    "flops_per_pass": gpt2_flops,
    "gflops_achieved": round(gpt2_flops / (gpt2_us * 1e-6) / 1e9, 2),
    "runs": 100
}

# ===== B) SHA-256 Python =====
print("\n=== SHA-256 Python (hashlib) ===")
data = b"A" * 32
iterations = 1000000

start = time.perf_counter()
for i in range(iterations):
    hashlib.sha256(data).digest()
end = time.perf_counter()

sha_total_s = end - start
sha_per_hash_us = (sha_total_s / iterations) * 1e6
sha_hps = iterations / sha_total_s

# Load ops from paso 3
with open("/home/kota/investigacion/fase1/sha256/mapa_sha256.json") as f:
    sha_data = json.load(f)
sha_ops = sha_data["total_por_bloque"]["total_ops"]

print(f"Per hash: {sha_per_hash_us:.2f} us")
print(f"Hashes/sec: {sha_hps:,.0f}")
print(f"Ops per hash: {sha_ops}")

results["sha256_python"] = {
    "per_hash_us": round(sha_per_hash_us, 3),
    "hashes_per_sec": round(sha_hps),
    "ops_per_hash": sha_ops,
    "iterations": iterations
}

# ===== C) SHA-256 C (compiled with OpenSSL) =====
print("\n=== SHA-256 C (OpenSSL) ===")
bench_dir = "/home/kota/investigacion/fase1/benchmark"
c_file = os.path.join(bench_dir, "sha256_bench.c")
bin_file = os.path.join(bench_dir, "sha256_bench")

# Compile
compile_cmd = f"gcc -O3 -march=native -o {bin_file} {c_file} -lcrypto"
print(f"Compiling: {compile_cmd}")
comp_result = subprocess.run(compile_cmd, shell=True, capture_output=True, text=True)

if comp_result.returncode != 0:
    print(f"Compile error: {comp_result.stderr}")
    results["sha256_c"] = {"error": comp_result.stderr}
else:
    # Run benchmark
    run_result = subprocess.run(bin_file, capture_output=True, text=True, timeout=120)
    print(run_result.stdout)

    # Parse output
    lines = run_result.stdout.strip().split('\n')
    c_time = float([l for l in lines if "Time:" in l][0].split(":")[1].strip().split()[0])
    c_hps = float([l for l in lines if "Speed:" in l][0].split(":")[1].strip().split()[0])
    c_per_hash_us = (c_time / 10000000) * 1e6

    results["sha256_c_openssl"] = {
        "per_hash_us": round(c_per_hash_us, 4),
        "hashes_per_sec": round(c_hps),
        "ops_per_hash": sha_ops,
        "iterations": 10000000,
        "compiler_flags": "-O3 -march=native"
    }

# ===== D) Ratios =====
print("\n=== Ratios ===")
sha256_per_forward = gpt2_us / sha_per_hash_us
print(f"1 GPT-2 forward = {sha256_per_forward:,.0f} SHA-256 (Python, tiempo)")

if "sha256_c_openssl" in results:
    c_per_hash_us = results["sha256_c_openssl"]["per_hash_us"]
    sha256_c_per_forward = gpt2_us / c_per_hash_us
    print(f"1 GPT-2 forward = {sha256_c_per_forward:,.0f} SHA-256 (C, tiempo)")

ops_ratio = gpt2_flops / sha_ops
print(f"1 GPT-2 forward = {ops_ratio:,.0f} SHA-256 (operaciones)")

results["ratios"] = {
    "gpt2_forward_us": round(gpt2_us, 1),
    "sha256_python_us": round(sha_per_hash_us, 3),
    "sha256_c_us": round(c_per_hash_us, 4) if "sha256_c_openssl" in results else None,
    "time_ratio_python": round(sha256_per_forward),
    "time_ratio_c": round(sha256_c_per_forward) if "sha256_c_openssl" in results else None,
    "ops_ratio": round(ops_ratio),
    "gpt2_flops": gpt2_flops,
    "sha256_ops": sha_ops
}

# Save
with open(os.path.join(bench_dir, "resultados.json"), "w") as f:
    json.dump(results, f, indent=2)

print(f"\nGuardado en resultados.json")
