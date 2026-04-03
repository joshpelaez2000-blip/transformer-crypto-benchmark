#!/usr/bin/env python3
"""Genera 50 preguntas de operaciones SHA-256 para benchmark de LLMs."""

import json
import random
import os

random.seed(42)  # Reproducibilidad

MOD32 = 2**32

def rotr32(val, n):
    """Rotación a la derecha en 32 bits."""
    return ((val >> n) | (val << (32 - n))) & 0xFFFFFFFF

preguntas = []

# 10 SUMA MODULAR 32 bits
for i in range(10):
    a = random.randint(0, MOD32 - 1)
    b = random.randint(0, MOD32 - 1)
    correct = (a + b) % MOD32
    preguntas.append({
        "id": i + 1,
        "categoria": "suma_mod32",
        "prompt": f"Calcula (A + B) mod 2^32 donde A={a} y B={b}. Responde SOLO el número, nada más.",
        "a": a, "b": b,
        "respuesta_correcta": correct
    })

# 10 XOR
for i in range(10):
    a = random.randint(0, MOD32 - 1)
    b = random.randint(0, MOD32 - 1)
    correct = a ^ b
    preguntas.append({
        "id": i + 11,
        "categoria": "xor",
        "prompt": f"Calcula A XOR B donde A={a} y B={b} (operación bit a bit). Responde SOLO el número decimal, nada más.",
        "a": a, "b": b,
        "respuesta_correcta": correct
    })

# 10 ROTACIÓN
for i in range(10):
    a = random.randint(0, MOD32 - 1)
    n = random.randint(1, 31)
    correct = rotr32(a, n)
    preguntas.append({
        "id": i + 21,
        "categoria": "rotacion",
        "prompt": f"Rota los bits del número A hacia la derecha N posiciones en un registro de 32 bits. A={a}, N={n}. Responde SOLO el número decimal resultado, nada más.",
        "a": a, "n": n,
        "respuesta_correcta": correct
    })

# 10 AND
for i in range(10):
    a = random.randint(0, MOD32 - 1)
    b = random.randint(0, MOD32 - 1)
    correct = a & b
    preguntas.append({
        "id": i + 31,
        "categoria": "and",
        "prompt": f"Calcula A AND B (operación bit a bit) donde A={a} y B={b}. Responde SOLO el número decimal, nada más.",
        "a": a, "b": b,
        "respuesta_correcta": correct
    })

# 10 COMBINADA
for i in range(10):
    a = random.randint(0, MOD32 - 1)
    b = random.randint(0, MOD32 - 1)
    c = random.randint(0, MOD32 - 1)
    correct = ((a + b) % MOD32) ^ c
    preguntas.append({
        "id": i + 41,
        "categoria": "combinada",
        "prompt": f"Calcula ((A + B) mod 4294967296) XOR C donde A={a}, B={b}, C={c}. Responde SOLO el número decimal, nada más.",
        "a": a, "b": b, "c": c,
        "respuesta_correcta": correct
    })

# Guardar
out_path = os.path.join(os.path.dirname(__file__), "preguntas_sha256.json")
with open(out_path, "w") as f:
    json.dump(preguntas, f, indent=2)

print(f"Generadas {len(preguntas)} preguntas en {out_path}")
for cat in ["suma_mod32", "xor", "rotacion", "and", "combinada"]:
    n = sum(1 for p in preguntas if p["categoria"] == cat)
    print(f"  {cat}: {n}")
