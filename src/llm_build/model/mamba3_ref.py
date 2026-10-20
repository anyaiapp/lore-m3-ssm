"""Selective scan for Mamba-style SSM (sequential reference + parallel scan for training)."""

from __future__ import annotations

import math

import torch


def _scan_coeffs(
    u: torch.Tensor,
    delta: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    dt = torch.sigmoid(delta)
    decay = torch.exp(-dt * torch.sigmoid(a))
    inp = dt * b * u
    return decay, inp


def selective_scan_sequential(
    u: torch.Tensor,
    delta: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
) -> torch.Tensor:
    """O(T) Python loop — reference only; use selective_scan() in training."""
    alpha, beta = _scan_coeffs(u, delta, a, b)
    bsz, seqlen, dim = u.shape
    y = torch.empty(bsz, seqlen, dim, device=u.device, dtype=u.dtype)
    h = torch.zeros(bsz, dim, device=u.device, dtype=u.dtype)
    for t in range(seqlen):
        h = alpha[:, t] * h + beta[:, t]
        y[:, t] = c[:, t] * h
    return y


def _associative_scan_1d(alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """
    Inclusive scan: h_t = alpha_t * h_{t-1} + beta_t, h_{-1} = 0.
    alpha, beta: (B, T, D). Returns h with same shape.
    Uses parallel prefix on (alpha, beta) pairs in O(log T) tensor steps.
    """
    a = alpha
    b = beta
    t_len = a.size(1)
    if t_len <= 1:
        return b
    levels = int(math.ceil(math.log2(t_len)))
    idx = torch.arange(t_len, device=a.device)
    for level in range(levels):
        shift = 1 << level
        if shift >= t_len:
            break
        a_left = torch.zeros_like(a)
        b_left = torch.zeros_like(b)
        a_left[:, shift:, :] = a[:, :-shift, :]
        b_left[:, shift:, :] = b[:, :-shift, :]
        a_new = a * a_left
        b_new = a * b_left + b
        mask = (idx >= shift).view(1, t_len, 1)
        a = torch.where(mask, a_new, a)
        b = torch.where(mask, b_new, b)
    return b


def selective_scan_parallel(
    u: torch.Tensor,
    delta: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
) -> torch.Tensor:
    """Vectorized parallel scan — default for training (all seq lengths, MPS/CUDA/CPU)."""
    alpha, beta = _scan_coeffs(u, delta, a, b)
    h = _associative_scan_1d(alpha, beta)
    return c * h


def selective_scan(
    u: torch.Tensor,
    delta: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    *,
    mode: str = "parallel",
) -> torch.Tensor:
    """
    u: (B, T, D)
    delta, a: (B, T, D)
    b, c: (B, T, D)
    Returns y: (B, T, D)
    """
    if mode == "sequential":
        return selective_scan_sequential(u, delta, a, b, c)
    return selective_scan_parallel(u, delta, a, b, c)
