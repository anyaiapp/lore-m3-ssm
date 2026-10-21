"""Selective scan for Mamba-style SSM (sequential, parallel, compiled)."""

from __future__ import annotations

import math
from functools import lru_cache

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
    """O(T) loop, vectorized over batch and channel — best backward on MPS for T≤2k."""
    alpha, beta = _scan_coeffs(u, delta, a, b)
    bsz, seqlen, dim = u.shape
    y = torch.empty(bsz, seqlen, dim, device=u.device, dtype=u.dtype)
    h = torch.zeros(bsz, dim, device=u.device, dtype=u.dtype)
    for t in range(seqlen):
        h = alpha[:, t] * h + beta[:, t]
        y[:, t] = c[:, t] * h
    return y


def _associative_scan_1d(alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """Blelloch-style parallel prefix for h_t = α_t h_{t-1} + β_t (same as sequential)."""
    a = alpha
    b = beta
    t_len = a.size(1)
    if t_len <= 1:
        return b
    levels = int(math.ceil(math.log2(t_len)))
    idx = torch.arange(t_len, device=a.device).view(1, t_len, 1)
    for level in range(levels):
        shift = 1 << level
        if shift >= t_len:
            break
        a_left = a.roll(shift, dims=1)
        b_left = b.roll(shift, dims=1)
        a_left[:, :shift, :] = 0
        b_left[:, :shift, :] = 0
        a_new = a * a_left
        b_new = a * b_left + b
        mask = idx >= shift
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
    """O(log T) depth, full (B,T,D) tensors — can be slower than sequential on MPS."""
    alpha, beta = _scan_coeffs(u, delta, a, b)
    h = _associative_scan_1d(alpha, beta)
    return c * h


@lru_cache(maxsize=2)
def _compiled_sequential_scan(mode: str):
    return torch.compile(selective_scan_sequential, mode=mode)


def selective_scan_compiled(
    u: torch.Tensor,
    delta: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
) -> torch.Tensor:
    if not hasattr(torch, "compile"):
        return selective_scan_sequential(u, delta, a, b, c)
    compile_mode = "max-autotune" if u.device.type == "cuda" else "reduce-overhead"
    return _compiled_sequential_scan(compile_mode)(u, delta, a, b, c)


def resolve_ssm_scan_mode(mode: str, seqlen: int, device: torch.device) -> str:
    if mode != "auto":
        return mode
    # On MPS, parallel scan does huge (B,T,D) ops per level; sequential wins.
    if device.type == "mps":
        return "sequential"
    if device.type == "cuda" and torch.cuda.is_available():
        from llm_build.model.mamba3_scan_triton import triton_scan_available

        if triton_scan_available():
            return "triton"
        if seqlen >= 2048:
            return "parallel"
        return "compiled"
    return "sequential"


def selective_scan(
    u: torch.Tensor,
    delta: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    *,
    mode: str = "auto",
    device: torch.device | None = None,
) -> torch.Tensor:
    dev = device if device is not None else u.device
    resolved = resolve_ssm_scan_mode(mode, u.size(1), dev)
    if resolved == "sequential":
        return selective_scan_sequential(u, delta, a, b, c)
    if resolved == "compiled":
        return selective_scan_compiled(u, delta, a, b, c)
    if resolved == "parallel":
        return selective_scan_parallel(u, delta, a, b, c)
    if resolved == "triton":
        from llm_build.model.mamba3_scan_triton import selective_scan_triton, triton_scan_available

        if triton_scan_available():
            return selective_scan_triton(u, delta, a, b, c)
        if u.size(1) <= 2048:
            return selective_scan_compiled(u, delta, a, b, c)
        return selective_scan_parallel(u, delta, a, b, c)
    raise ValueError(f"Unknown ssm_scan mode: {mode} (resolved={resolved})")
