#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from llm_build.config import load_config
from llm_build.data.memmap_dataset import MemmapTokenDataset
from llm_build.data.streaming_dataset import build_streaming_loader
from llm_build.model.lore_m3 import LoreM3
from llm_build.train.trainer import train_loop


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/experiment_100mb.yaml")
    args = p.parse_args()
    cfg = load_config(args.config)
    random.seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    torch.manual_seed(cfg.train.seed)

    val_loader = None
    if cfg.data.streaming:
        train_loader = build_streaming_loader(cfg)
        tokens_val = Path(cfg.paths.tokens_dir) / "val.bin"
        if tokens_val.exists():
            val_ds = MemmapTokenDataset(cfg.paths.tokens_dir, "val", seq_len=cfg.train.seq_len)
            val_loader = DataLoader(val_ds, batch_size=cfg.train.micro_batch, shuffle=False, drop_last=True)
    else:
        train_ds = MemmapTokenDataset(cfg.paths.tokens_dir, "train", seq_len=cfg.train.seq_len)
        val_ds = MemmapTokenDataset(cfg.paths.tokens_dir, "val", seq_len=cfg.train.seq_len)
        train_loader = DataLoader(train_ds, batch_size=cfg.train.micro_batch, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=cfg.train.micro_batch, shuffle=False, drop_last=True)

    model = LoreM3(cfg.model, cfg.tokenizer.vocab_size)
    try:
        import mlflow

        mlflow.set_experiment(cfg.train.mlflow_experiment)
        mlflow.start_run(run_name="lore-m3-train")
    except Exception as e:
        print(f"MLflow unavailable: {e}")
    train_loop(cfg, model, train_loader, val_loader)


if __name__ == "__main__":
    main()
