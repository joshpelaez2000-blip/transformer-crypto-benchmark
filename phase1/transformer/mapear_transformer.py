#!/usr/bin/env python3
"""Mapea TODAS las operaciones de un forward pass de GPT-2 small (seq_len=1)."""

import json
from transformers import GPT2Model

model = GPT2Model.from_pretrained('gpt2')
config = model.config

# GPT-2 small params
n_layer = config.n_layer        # 12
n_head = config.n_head          # 12
d_model = config.n_embd         # 768
d_head = d_model // n_head      # 64
d_ff = config.n_inner or 4 * d_model  # 3072
vocab_size = config.vocab_size  # 50257
seq_len = 1  # single token forward pass

capas = []
total_mul = 0
total_sum = 0
total_exp = 0
total_div = 0
total_sqrt = 0
total_params = 0

# 1. Token Embedding (lookup, no math ops)
p = vocab_size * d_model
capas.append({
    "nombre": "wte (token embedding)",
    "tipo": "embedding",
    "shape_pesos": [vocab_size, d_model],
    "params": p,
    "ops": {"mul_float32": 0, "sum_float32": 0, "exp": 0, "div": 0, "sqrt": 0}
})
total_params += p

# 2. Position Embedding (lookup, no math ops)
max_pos = config.n_positions  # 1024
p = max_pos * d_model
capas.append({
    "nombre": "wpe (position embedding)",
    "tipo": "embedding",
    "shape_pesos": [max_pos, d_model],
    "params": p,
    "ops": {"mul_float32": 0, "sum_float32": 0, "exp": 0, "div": 0, "sqrt": 0}
})
total_params += p

# 3. Embedding sum (token + position)
# seq_len * d_model additions
add_ops = seq_len * d_model
capas.append({
    "nombre": "embedding_sum (token + position)",
    "tipo": "addition",
    "shape_pesos": [],
    "params": 0,
    "ops": {"mul_float32": 0, "sum_float32": add_ops, "exp": 0, "div": 0, "sqrt": 0}
})
total_sum += add_ops

# 4. Each transformer block (x12)
for i in range(n_layer):
    block_capas = []

    # 4a. LayerNorm 1 (before attention)
    # LN: mean (d_model-1 sums + 1 div), variance (d_model muls + d_model-1 sums + 1 div),
    #     normalize (d_model subs + d_model divs + d_model sqrts... simplified:)
    # Per token: mean = d_model-1 sums + 1 div, var = d_model muls + d_model-1 sums + 1 div
    # normalize: d_model subs(=sums) + d_model divs, scale: d_model muls, shift: d_model sums
    # sqrt of variance: 1
    ln_mul = seq_len * (d_model + d_model)  # variance muls + gamma scale
    ln_sum = seq_len * ((d_model - 1) + (d_model - 1) + d_model + d_model)  # mean + var + sub + beta
    ln_div = seq_len * (1 + 1 + d_model)  # mean_div + var_div + normalize_div
    ln_sqrt = seq_len * 1
    ln_params = 2 * d_model  # gamma + beta

    block_capas.append({
        "nombre": f"h.{i}.ln_1 (layernorm)",
        "tipo": "layernorm",
        "shape_pesos": [d_model],
        "params": ln_params,
        "ops": {"mul_float32": ln_mul, "sum_float32": ln_sum, "exp": 0, "div": ln_div, "sqrt": ln_sqrt}
    })

    # 4b. Attention: Q, K, V projections
    # Each is matmul [seq_len, d_model] x [d_model, d_model] + bias
    # muls = seq_len * d_model * d_model, sums = seq_len * (d_model-1) * d_model + seq_len * d_model (bias)
    for name in ["q_proj", "k_proj", "v_proj"]:
        m = seq_len * d_model * d_model
        s = seq_len * (d_model - 1) * d_model + seq_len * d_model  # matmul sums + bias add
        p = d_model * d_model + d_model  # weight + bias
        block_capas.append({
            "nombre": f"h.{i}.attn.{name}",
            "tipo": "attention_projection",
            "shape_pesos": [d_model, d_model],
            "params": p,
            "ops": {"mul_float32": m, "sum_float32": s, "exp": 0, "div": 0, "sqrt": 0}
        })

    # 4c. Attention scores: Q @ K^T / sqrt(d_head), per head
    # Each head: [seq_len, d_head] x [d_head, seq_len] = seq_len * d_head * seq_len muls
    # With seq_len=1: n_head * (1 * d_head * 1) muls, n_head * (1 * (d_head-1) * 1) sums
    # Division by sqrt(d_head): n_head * seq_len * seq_len divs
    # sqrt(d_head): 1 (constant, but counting it)
    attn_score_mul = n_head * seq_len * d_head * seq_len
    attn_score_sum = n_head * seq_len * (d_head - 1) * seq_len
    attn_score_div = n_head * seq_len * seq_len
    attn_score_sqrt = 1  # sqrt(d_head) computed once

    block_capas.append({
        "nombre": f"h.{i}.attn.scores (Q@K^T/sqrt(dk))",
        "tipo": "attention_scores",
        "shape_pesos": [],
        "params": 0,
        "ops": {"mul_float32": attn_score_mul, "sum_float32": attn_score_sum, "exp": 0, "div": attn_score_div, "sqrt": attn_score_sqrt}
    })

    # 4d. Softmax over attention scores
    # Per head: exp(seq_len values) + sum(seq_len exps) + div(seq_len values)
    # With seq_len=1: n_head * 1 exp, n_head * 0 sums (single element), n_head * 1 div
    attn_soft_exp = n_head * seq_len * seq_len  # exp of each score
    attn_soft_sum = n_head * (seq_len * seq_len - 1)  # sum for denominator (0 when seq_len=1)
    attn_soft_div = n_head * seq_len * seq_len  # divide each by sum

    block_capas.append({
        "nombre": f"h.{i}.attn.softmax",
        "tipo": "softmax",
        "shape_pesos": [],
        "params": 0,
        "ops": {"mul_float32": 0, "sum_float32": attn_soft_sum, "exp": attn_soft_exp, "div": attn_soft_div, "sqrt": 0}
    })

    # 4e. Attention output: attn_weights @ V, per head
    # Each head: [seq_len, seq_len] x [seq_len, d_head]
    # muls = seq_len * seq_len * d_head, sums = seq_len * (seq_len-1) * d_head
    attn_v_mul = n_head * seq_len * seq_len * d_head
    attn_v_sum = n_head * seq_len * (seq_len - 1) * d_head  # 0 when seq_len=1

    block_capas.append({
        "nombre": f"h.{i}.attn.output (weights@V)",
        "tipo": "attention_output",
        "shape_pesos": [],
        "params": 0,
        "ops": {"mul_float32": attn_v_mul, "sum_float32": attn_v_sum, "exp": 0, "div": 0, "sqrt": 0}
    })

    # 4f. Output projection: [seq_len, d_model] x [d_model, d_model] + bias
    m = seq_len * d_model * d_model
    s = seq_len * (d_model - 1) * d_model + seq_len * d_model
    p = d_model * d_model + d_model
    block_capas.append({
        "nombre": f"h.{i}.attn.c_proj (output projection)",
        "tipo": "attention_projection",
        "shape_pesos": [d_model, d_model],
        "params": p,
        "ops": {"mul_float32": m, "sum_float32": s, "exp": 0, "div": 0, "sqrt": 0}
    })

    # 4g. Residual connection (add)
    res1_sum = seq_len * d_model
    block_capas.append({
        "nombre": f"h.{i}.residual_1",
        "tipo": "residual",
        "shape_pesos": [],
        "params": 0,
        "ops": {"mul_float32": 0, "sum_float32": res1_sum, "exp": 0, "div": 0, "sqrt": 0}
    })

    # 4h. LayerNorm 2 (before FFN) - same as LN1
    block_capas.append({
        "nombre": f"h.{i}.ln_2 (layernorm)",
        "tipo": "layernorm",
        "shape_pesos": [d_model],
        "params": ln_params,
        "ops": {"mul_float32": ln_mul, "sum_float32": ln_sum, "exp": 0, "div": ln_div, "sqrt": ln_sqrt}
    })

    # 4i. FFN layer 1: [seq_len, d_model] x [d_model, d_ff] + bias
    m = seq_len * d_model * d_ff
    s = seq_len * (d_model - 1) * d_ff + seq_len * d_ff
    p = d_model * d_ff + d_ff
    block_capas.append({
        "nombre": f"h.{i}.mlp.c_fc (FFN up)",
        "tipo": "feedforward",
        "shape_pesos": [d_model, d_ff],
        "params": p,
        "ops": {"mul_float32": m, "sum_float32": s, "exp": 0, "div": 0, "sqrt": 0}
    })

    # 4j. GELU activation
    # GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    # Per element: ~1 mul(x^3) + 1 mul(0.044715*) + 1 sum(x+) + 1 mul(sqrt(2/pi)*) +
    #   1 tanh(~= exp+div+sum) + 1 sum(1+) + 1 mul(x*) + 1 mul(0.5*)
    # Simplified count per element: 4 muls, 2 sums, 1 exp, 1 div
    gelu_elements = seq_len * d_ff
    block_capas.append({
        "nombre": f"h.{i}.mlp.gelu",
        "tipo": "activation",
        "shape_pesos": [],
        "params": 0,
        "ops": {
            "mul_float32": 4 * gelu_elements,
            "sum_float32": 2 * gelu_elements,
            "exp": gelu_elements,  # inside tanh
            "div": gelu_elements,  # inside tanh
            "sqrt": 0
        }
    })

    # 4k. FFN layer 2: [seq_len, d_ff] x [d_ff, d_model] + bias
    m = seq_len * d_ff * d_model
    s = seq_len * (d_ff - 1) * d_model + seq_len * d_model
    p = d_ff * d_model + d_model
    block_capas.append({
        "nombre": f"h.{i}.mlp.c_proj (FFN down)",
        "tipo": "feedforward",
        "shape_pesos": [d_ff, d_model],
        "params": p,
        "ops": {"mul_float32": m, "sum_float32": s, "exp": 0, "div": 0, "sqrt": 0}
    })

    # 4l. Residual connection 2
    block_capas.append({
        "nombre": f"h.{i}.residual_2",
        "tipo": "residual",
        "shape_pesos": [],
        "params": 0,
        "ops": {"mul_float32": 0, "sum_float32": res1_sum, "exp": 0, "div": 0, "sqrt": 0}
    })

    for c in block_capas:
        total_mul += c["ops"]["mul_float32"]
        total_sum += c["ops"]["sum_float32"]
        total_exp += c["ops"]["exp"]
        total_div += c["ops"]["div"]
        total_sqrt += c["ops"]["sqrt"]
        total_params += c["params"]
        capas.append(c)

# 5. Final LayerNorm
capas.append({
    "nombre": "ln_f (final layernorm)",
    "tipo": "layernorm",
    "shape_pesos": [d_model],
    "params": 2 * d_model,
    "ops": {"mul_float32": ln_mul, "sum_float32": ln_sum, "exp": 0, "div": ln_div, "sqrt": ln_sqrt}
})
total_mul += ln_mul
total_sum += ln_sum
total_div += ln_div
total_sqrt += ln_sqrt
total_params += 2 * d_model

total_flops = total_mul + total_sum  # FLOPs = muls + adds typically

result = {
    "modelo": "gpt2",
    "config": {
        "n_layer": n_layer,
        "n_head": n_head,
        "d_model": d_model,
        "d_head": d_head,
        "d_ff": d_ff,
        "vocab_size": vocab_size,
        "seq_len": seq_len
    },
    "params_total": total_params,
    "capas": capas,
    "total_ops_por_token": {
        "mul_float32": total_mul,
        "sum_float32": total_sum,
        "exp": total_exp,
        "div": total_div,
        "sqrt": total_sqrt,
        "total_flops": total_flops
    }
}

# Verify param count against model
model_params = sum(p.numel() for p in model.parameters())
result["params_verificacion"] = {
    "calculado": total_params,
    "pytorch": model_params,
    "match": total_params == model_params
}

with open("/home/kota/investigacion/fase1/transformer/mapa_operaciones.json", "w") as f:
    json.dump(result, f, indent=2)

print(f"Params calculado: {total_params:,}")
print(f"Params PyTorch:   {model_params:,}")
print(f"Match: {total_params == model_params}")
print(f"\nOps por token (seq_len=1):")
print(f"  mul_float32: {total_mul:,}")
print(f"  sum_float32: {total_sum:,}")
print(f"  exp:         {total_exp:,}")
print(f"  div:         {total_div:,}")
print(f"  sqrt:        {total_sqrt:,}")
print(f"  total_flops: {total_flops:,}")
print(f"\nGuardado en mapa_operaciones.json")
