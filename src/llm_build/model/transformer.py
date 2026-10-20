from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from llm_build.model.mlp import SwiGLU
from llm_build.model.norms import RMSNorm
from llm_build.model.rope import RotaryEmbedding, apply_rope


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        max_seq_len: int,
        attn_window: int | None,
        qk_norm: bool,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.attn_window = attn_window
        self.q_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, d_model, bias=False)
        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.rope = RotaryEmbedding(head_dim, max_seq_len)
        self._cached_mask_key: tuple[int, int, torch.device] | None = None
        self._cached_attn_mask: torch.Tensor | None = None

    def _sliding_mask(self, t: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
        if self.attn_window is None or t <= 1:
            return None
        win = min(self.attn_window, t)
        key = (t, win, device)
        if self._cached_mask_key == key and self._cached_attn_mask is not None:
            return self._cached_attn_mask.to(dtype=dtype)
        rows = torch.arange(t, device=device)
        cols = torch.arange(t, device=device)
        dist = rows.unsqueeze(1) - cols.unsqueeze(0)
        allowed = (dist >= 0) & (dist < win)
        mask = torch.zeros(t, t, device=device, dtype=dtype)
        mask.masked_fill_(~allowed, float("-inf"))
        self._cached_mask_key = key
        self._cached_attn_mask = mask
        return mask

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_kv_heads == self.n_heads:
            return x
        reps = self.n_heads // self.n_kv_heads
        b, t, h, d = x.shape
        x = x.unsqueeze(3).expand(b, t, h, reps, d)
        return x.reshape(b, t, self.n_heads, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(b, t, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(b, t, self.n_kv_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        cos, sin = self.rope(t, x.device)
        q, k = apply_rope(q, k, cos, sin)
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn_mask = self._sliding_mask(t, x.device, q.dtype)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=attn_mask is None,
            dropout_p=0.0,
        )
        out = out.transpose(1, 2).contiguous().view(b, t, self.n_heads * self.head_dim)
        return self.o_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_kv_heads=cfg.n_kv_heads,
            head_dim=cfg.head_dim,
            max_seq_len=cfg.max_seq_len,
            attn_window=cfg.attn_window,
            qk_norm=cfg.qk_norm,
        )
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg.d_model, cfg.mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
