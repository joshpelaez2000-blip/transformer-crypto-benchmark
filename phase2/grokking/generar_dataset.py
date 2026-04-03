#!/usr/bin/env python3
"""Genera datasets para Grokking: suma modular y XOR modular."""

import json
import random

P = 97  # primo, como en el paper de Grokking

random.seed(42)

# Generar TODAS las combinaciones
all_pairs = [(a, b) for a in range(P) for b in range(P)]
random.shuffle(all_pairs)

# Split 50/50
mid = len(all_pairs) // 2
train_pairs = all_pairs[:mid]
test_pairs = all_pairs[mid:]

# Dataset 1: (a + b) mod p
dataset_add = {
    "operation": "mod_add",
    "p": P,
    "train": [{"a": a, "b": b, "target": (a + b) % P} for a, b in train_pairs],
    "test": [{"a": a, "b": b, "target": (a + b) % P} for a, b in test_pairs],
}

# Dataset 2: (a XOR b) mod p
dataset_xor = {
    "operation": "mod_xor",
    "p": P,
    "train": [{"a": a, "b": b, "target": (a ^ b) % P} for a, b in train_pairs],
    "test": [{"a": a, "b": b, "target": (a ^ b) % P} for a, b in test_pairs],
}

with open("/home/kota/investigacion/fase2/grokking/dataset_mod_add.json", "w") as f:
    json.dump(dataset_add, f)

with open("/home/kota/investigacion/fase2/grokking/dataset_mod_xor.json", "w") as f:
    json.dump(dataset_xor, f)

print(f"P = {P}")
print(f"Total pares: {len(all_pairs)}")
print(f"Train: {len(train_pairs)}, Test: {len(test_pairs)}")
print(f"Ejemplo add: {train_pairs[0]} -> {(train_pairs[0][0] + train_pairs[0][1]) % P}")
print(f"Ejemplo xor: {train_pairs[0]} -> {(train_pairs[0][0] ^ train_pairs[0][1]) % P}")
print("Datasets guardados.")
