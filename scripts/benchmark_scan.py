#!/usr/bin/env python3
"""Benchmark selective-scan implementations (forward + backward)."""
from __future__ import annotations

import argparse
import time

import torch

from llm_build.model.mamba3_ref import (
    resolve_ssm_scan_mode,
    selective_scan_compiled,
    selective_scan_parallel,
    selective_scan_sequential,
)
from llm_build.model.mamba3_scan_triton import selective_scan_triton, triton_scan_available
from llm_build.train.trainer import resolve_device


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _bench(
    fn,
    tensors: tuple[torch.Tensor, ...],
    device: torch.device,
    warmup: int,
    reps: int,
) -> float:
    for _ in range(warmup):
        y = fn(*tensors)
        y.sum().backward()
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(reps):
        for t in tensors:
            if t.grad is not None:
                t.grad = None
        y = fn(*tensors)
        y.sum().backward()
    _sync(device)
    return (time.perf_counter() - t0) / reps


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--seqlen", type=int, default=1024)
    p.add_argument("--dim", type=int, default=1024)
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    device = resolve_device(args.device)
    b, t, d = args.batch, args.seqlen, args.dim
    u = torch.randn(b, t, d, device=device, requires_grad=True)
    delta = torch.randn(b, t, d, device=device, requires_grad=True)
    a = torch.randn(b, t, d, device=device, requires_grad=True)
    bvec = torch.randn(b, t, d, device=device, requires_grad=True)
    c = torch.randn(b, t, d, device=device, requires_grad=True)
    tensors = (u, delta, a, bvec, c)

    modes: list[tuple[str, object]] = [
        ("sequential", selective_scan_sequential),
        ("parallel", selective_scan_parallel),
    ]
    if device.type == "cuda":
        modes.append(("compiled", selective_scan_compiled))
        if triton_scan_available():
            modes.append(("triton", selective_scan_triton))

    print(f"device={device} shape B={b} T={t} D={d} (fwd+bwd per call)")
    for name, fn in modes:
        u.grad = delta.grad = a.grad = bvec.grad = c.grad = None
        try:
            sec = _bench(fn, tensors, device, warmup=3, reps=args.reps)
        except Exception as exc:
            print(f"  {name:12s} skipped ({exc})")
            continue
        print(f"  {name:12s} {sec * 1000:8.2f} ms")

    mode = resolve_ssm_scan_mode("auto", t, device)
    print(f"  ssm_scan auto → {mode}")


if __name__ == "__main__":
    main()
