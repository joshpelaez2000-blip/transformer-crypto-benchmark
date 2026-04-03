#!/usr/bin/env python3
"""SHA-256 manual, ronda por ronda, contando CADA operación."""

import json
import hashlib
import struct

# SHA-256 constants
K = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
]

H0 = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
]

MASK32 = 0xFFFFFFFF

# Operation counters
ops = {
    "sumas_mod32": 0,
    "rotaciones": 0,
    "shifts": 0,
    "xor": 0,
    "and": 0,
    "not": 0
}

def rotr(x, n):
    ops["rotaciones"] += 1
    return ((x >> n) | (x << (32 - n))) & MASK32

def shr(x, n):
    ops["shifts"] += 1
    return x >> n

def add32(*args):
    result = args[0]
    for a in args[1:]:
        ops["sumas_mod32"] += 1
        result = (result + a) & MASK32
    return result

def xor(*args):
    result = args[0]
    for a in args[1:]:
        ops["xor"] += 1
        result = result ^ a
    return result

def and_op(a, b):
    ops["and"] += 1
    return a & b

def not_op(a):
    ops["not"] += 1
    return a ^ MASK32

def Sigma0(x):
    return xor(rotr(x, 2), rotr(x, 13), rotr(x, 22))

def Sigma1(x):
    return xor(rotr(x, 6), rotr(x, 11), rotr(x, 25))

def sigma0(x):
    return xor(rotr(x, 7), rotr(x, 18), shr(x, 3))

def sigma1(x):
    return xor(rotr(x, 17), rotr(x, 19), shr(x, 10))

def Ch(e, f, g):
    # (e AND f) XOR (NOT e AND g)
    return xor(and_op(e, f), and_op(not_op(e), g))

def Maj(a, b, c):
    # (a AND b) XOR (a AND c) XOR (b AND c)
    return xor(and_op(a, b), and_op(a, c), and_op(b, c))

def pad_message(message_bytes):
    """SHA-256 padding"""
    msg_len = len(message_bytes)
    bit_len = msg_len * 8
    # Append 1 bit + zeros
    message_bytes += b'\x80'
    while (len(message_bytes) % 64) != 56:
        message_bytes += b'\x00'
    # Append original length as 64-bit big-endian
    message_bytes += struct.pack('>Q', bit_len)
    return message_bytes

def sha256_manual(message_bytes):
    """Full SHA-256 with operation counting."""
    global ops
    ops = {"sumas_mod32": 0, "rotaciones": 0, "shifts": 0, "xor": 0, "and": 0, "not": 0}

    padded = pad_message(bytearray(message_bytes))
    assert len(padded) % 64 == 0

    h = list(H0)
    num_blocks = len(padded) // 64

    for block_idx in range(num_blocks):
        block = padded[block_idx * 64:(block_idx + 1) * 64]

        # Parse block into 16 32-bit words
        w = list(struct.unpack('>16I', block))

        # Message schedule: extend to 64 words
        for i in range(16, 64):
            w.append(add32(sigma1(w[i-2]), w[i-7], sigma0(w[i-15]), w[i-16]))

        # Initialize working variables
        a, b, c, d, e, f, g, hh = h

        # Track per-round ops
        round_ops_before = dict(ops)

        # 64 compression rounds
        for i in range(64):
            T1 = add32(hh, Sigma1(e), Ch(e, f, g), K[i], w[i])
            T2 = add32(Sigma0(a), Maj(a, b, c))

            hh = g
            g = f
            f = e
            e = add32(d, T1)
            d = c
            c = b
            b = a
            a = add32(T1, T2)

        # Add compressed chunk to hash value
        h[0] = add32(h[0], a)
        h[1] = add32(h[1], b)
        h[2] = add32(h[2], c)
        h[3] = add32(h[3], d)
        h[4] = add32(h[4], e)
        h[5] = add32(h[5], f)
        h[6] = add32(h[6], g)
        h[7] = add32(h[7], hh)

    return struct.pack('>8I', *h).hex()


# Test with known input
test_input = b"abc"
manual_hash = sha256_manual(test_input)
expected_hash = hashlib.sha256(test_input).hexdigest()

print(f"Manual:   {manual_hash}")
print(f"Expected: {expected_hash}")
print(f"Match: {manual_hash == expected_hash}")

# Now count per-round (reset and do single block with tracking)
# For per-round analysis, process single block and track per round
ops_reset = {"sumas_mod32": 0, "rotaciones": 0, "shifts": 0, "xor": 0, "and": 0, "not": 0}

# Run again to get total for 1 block (abc = 1 block after padding)
final_hash = sha256_manual(test_input)
total_ops = dict(ops)

# Calculate per-round (message schedule + 64 rounds)
# Message schedule ops (rounds 16-63, 48 iterations):
#   Each: sigma1 (2 rotr + 1 shr + 2 xor) + sigma0 (2 rotr + 1 shr + 2 xor) + add32(4 args = 3 sums)
# = 48 * (4 rotr + 2 shr + 4 xor + 3 sums) = 192 rotr + 96 shr + 192 xor + 144 sums
msg_sched_ops = {
    "sumas_mod32": 48 * 3,  # 144
    "rotaciones": 48 * 4,   # 192
    "shifts": 48 * 2,       # 96
    "xor": 48 * 4,          # 192
    "and": 0,
    "not": 0
}

# Per compression round:
# Sigma1(e): 3 rotr + 2 xor
# Ch(e,f,g): 1 not + 2 and + 1 xor
# T1 = add32(hh, Sigma1, Ch, K[i], w[i]): 4 sums
# Sigma0(a): 3 rotr + 2 xor
# Maj(a,b,c): 3 and + 2 xor
# T2 = add32(Sigma0, Maj): 1 sum
# e = add32(d, T1): 1 sum
# a = add32(T1, T2): 1 sum
per_round = {
    "sumas_mod32": 7,   # 4 + 1 + 1 + 1
    "rotaciones": 6,    # 3 + 3
    "shifts": 0,
    "xor": 7,           # 2 + 1 + 2 + 2
    "and": 5,           # 2 + 3
    "not": 1
}

# Final addition: 8 add32 = 8 sums
final_add = {"sumas_mod32": 8}

# Total per block (calculated)
calc_total = {}
for key in per_round:
    calc_total[key] = msg_sched_ops.get(key, 0) + per_round[key] * 64 + (final_add.get(key, 0))

total_all_ops = sum(total_ops.values())
calc_all_ops = sum(calc_total.values())

print(f"\nOperaciones contadas (1 bloque):")
for k, v in total_ops.items():
    print(f"  {k}: {v}")
print(f"  TOTAL: {total_all_ops}")

print(f"\nOperaciones calculadas teóricamente:")
for k, v in calc_total.items():
    print(f"  {k}: {v}")
print(f"  TOTAL: {calc_all_ops}")

# Save results
result = {
    "algoritmo": "SHA-256",
    "input_test": "abc",
    "por_ronda_compresion": per_round,
    "message_schedule_ops": msg_sched_ops,
    "final_addition_ops": {"sumas_mod32": 8},
    "total_rondas": 64,
    "total_por_bloque": {
        "sumas_mod32": total_ops["sumas_mod32"],
        "rotaciones": total_ops["rotaciones"],
        "shifts": total_ops["shifts"],
        "xor": total_ops["xor"],
        "and": total_ops["and"],
        "not": total_ops["not"],
        "total_ops": total_all_ops
    },
    "verificacion": f"{'PASS' if manual_hash == expected_hash else 'FAIL'} — manual={manual_hash} expected={expected_hash}"
}

with open("/home/kota/investigacion/fase1/sha256/mapa_sha256.json", "w") as f:
    json.dump(result, f, indent=2)

print(f"\nGuardado en mapa_sha256.json")
