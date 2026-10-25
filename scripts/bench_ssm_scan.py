#!/usr/bin/env python3
"""Benchmark SSM scan modes and (optionally) full-model tok/s before a long train run."""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import torch

from llm_build.config import load_config
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


def _bench_scan(
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
    p.add_argument("--config", default="configs/experiment_100mb.yaml")
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument(
        "--with-step",
        action="store_true",
        help="After scan bench, run scripts/benchmark_step.py on the same config",
    )
    args = p.parse_args()
    cfg = load_config(args.config)
    device = resolve_device(cfg.train.device)
    b = cfg.train.micro_batch
    t_len = cfg.train.seq_len
    d = cfg.model.d_model * cfg.model.expand

    u = torch.randn(b, t_len, d, device=device, requires_grad=True)
    delta = torch.randn(b, t_len, d, device=device, requires_grad=True)
    a = torch.randn(b, t_len, d, device=device, requires_grad=True)
    bvec = torch.randn(b, t_len, d, device=device, requires_grad=True)
    c = torch.randn(b, t_len, d, device=device, requires_grad=True)
    tensors = (u, delta, a, bvec, c)

    modes: list[tuple[str, object]] = [
        ("sequential", selective_scan_sequential),
        ("parallel", selective_scan_parallel),
    ]
    if device.type == "cuda":
        modes.append(("compiled", selective_scan_compiled))
        if triton_scan_available():
            modes.append(("triton", selective_scan_triton))

    auto = resolve_ssm_scan_mode(cfg.model.ssm_scan, t_len, device)
    print(f"device={device} B={b} T={t_len} D={d} (Mamba d_inner) | ssm_scan auto → {auto}")
    print("selective scan (fwd+bwd per call):")
    for name, fn in modes:
        for tensor in tensors:
            tensor.grad = None
        try:
            sec = _bench_scan(fn, tensors, device, args.warmup, args.reps)
        except Exception as exc:
            msg = str(exc).splitlines()[0][:120]
            print(f"  {name:12s} skipped ({type(exc).__name__}: {msg})")
            continue
        print(f"  {name:12s} {sec * 1000:8.2f} ms")

    if args.with_step:
        step_script = Path(__file__).resolve().parent / "benchmark_step.py"
        print()
        subprocess.run(
            [sys.executable, str(step_script), "--config", args.config],
            check=True,
        )


if __name__ == "__main__":
    main()
