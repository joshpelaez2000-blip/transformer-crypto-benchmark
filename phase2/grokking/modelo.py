#!/usr/bin/env python3
"""Transformer mínimo de 2 capas para Grokking. Desde cero con PyTorch."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight

class SelfAttention(nn.Module):
    def __init__(self, dim, n_heads=1):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)  # B, H, T, D
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) / self.scale
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out(out)

class FFN(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)))

class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, ff_dim):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = SelfAttention(dim, n_heads)
        self.norm2 = RMSNorm(dim)
        self.ffn = FFN(dim, ff_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class GrokTransformer(nn.Module):
    def __init__(self, p=97, dim=128, n_layers=2, n_heads=1, ff_dim=512):
        super().__init__()
        self.p = p
        self.embed_a = nn.Embedding(p, dim)
        self.embed_b = nn.Embedding(p, dim)
        self.embed_op = nn.Embedding(1, dim)  # token for the operation
        self.pos_embed = nn.Embedding(3, dim)  # 3 positions: a, op, b

        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_heads, ff_dim)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(dim)
        self.head = nn.Linear(dim, p, bias=False)

    def forward(self, a, b):
        # a, b: (B,) integers in [0, p-1]
        B = a.shape[0]
        pos = torch.arange(3, device=a.device).unsqueeze(0).expand(B, -1)

        # Tokens: [a, op, b]
        tok_a = self.embed_a(a).unsqueeze(1)
        tok_op = self.embed_op(torch.zeros(B, dtype=torch.long, device=a.device)).unsqueeze(1)
        tok_b = self.embed_b(b).unsqueeze(1)

        x = torch.cat([tok_a, tok_op, tok_b], dim=1)  # B, 3, dim
        x = x + self.pos_embed(pos)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        # Use the last token for prediction
        logits = self.head(x[:, -1, :])  # B, p
        return logits

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

if __name__ == "__main__":
    model = GrokTransformer(p=97, dim=128, n_layers=2, n_heads=1, ff_dim=512)
    print(f"Parameters: {model.count_params():,}")
    a = torch.tensor([5, 10])
    b = torch.tensor([3, 20])
    logits = model(a, b)
    print(f"Output shape: {logits.shape}")  # (2, 97)
