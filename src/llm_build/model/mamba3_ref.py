"""PyTorch reference selective scan for Mamba-style SSM (MPS-safe)."""

from __future__ import annotations

import torch


def selective_scan(
    u: torch.Tensor,
    delta: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
) -> torch.Tensor:
    """
    u: (B, T, D)
    delta, a: (B, T, D)
    b, c: (B, T, D)
    Returns y: (B, T, D)
    """
    bsz, seqlen, dim = u.shape
    h = torch.zeros(bsz, dim, device=u.device, dtype=u.dtype)
    ys = []
    for t in range(seqlen):
        dt = torch.sigmoid(delta[:, t])
        decay = torch.exp(-dt * torch.sigmoid(a[:, t]))
        inp = dt * b[:, t] * u[:, t]
        h = decay * h + inp
        ys.append(c[:, t] * h)
    return torch.stack(ys, dim=1)
