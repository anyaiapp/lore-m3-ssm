#!/usr/bin/env python3
"""Time one optimizer step (grad_accum micro-batches) for tuning batch size on CUDA/MPS."""
from __future__ import annotations

import argparse
import time

import torch
from torch.utils.data import DataLoader

from llm_build.config import load_config
from llm_build.data.memmap_dataset import MemmapTokenDataset
from llm_build.model.lore_m3 import LoreM3
from llm_build.train.runtime import apply_model_runtime_opts
from llm_build.train.trainer import resolve_device


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/experiment_100mb.yaml")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument(
        "--no-grad-checkpoint",
        action="store_true",
        help="Disable activation checkpointing (faster if memory allows)",
    )
    args = p.parse_args()
    cfg = load_config(args.config)
    if args.no_grad_checkpoint:
        cfg.train.grad_checkpoint = False
    device = resolve_device(cfg.train.device)
    ds = MemmapTokenDataset(cfg.paths.tokens_dir, "train", seq_len=cfg.train.seq_len)
    loader = DataLoader(
        ds,
        batch_size=cfg.train.micro_batch,
        shuffle=True,
        drop_last=True,
        pin_memory=device.type == "cuda",
    )
    model = LoreM3(cfg.model, cfg.tokenizer.vocab_size, grad_checkpoint=cfg.train.grad_checkpoint)
    model.to(device)
    for line in apply_model_runtime_opts(model, cfg, device):
        print(line)
    model.train()
    use_amp = cfg.train.amp and device.type in ("cuda", "mps")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)
    it = iter(loader)

    def one_step() -> float:
        opt.zero_grad(set_to_none=True)
        t0 = time.perf_counter()
        for _ in range(cfg.train.grad_accum):
            batch = next(it)
            ids = batch["input_ids"].to(device, non_blocking=device.type == "cuda")
            labels = batch["labels"].to(device, non_blocking=device.type == "cuda")
            with torch.autocast(device_type=device.type, enabled=use_amp):
                _, loss = model(ids, labels)
            (loss / cfg.train.grad_accum).backward()
        opt.step()
        return time.perf_counter() - t0

    for _ in range(args.warmup):
        one_step()
    times = [one_step() for _ in range(args.steps)]
    avg = sum(times) / len(times)
    tok = cfg.train.micro_batch * cfg.train.seq_len * cfg.train.grad_accum
    print(
        f"device={device} micro_batch={cfg.train.micro_batch} grad_accum={cfg.train.grad_accum} "
        f"grad_checkpoint={cfg.train.grad_checkpoint}"
    )
    print(f"avg step: {avg:.2f}s | tok/s: {tok / avg:,.0f}")


if __name__ == "__main__":
    main()
