#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from llm_build.config import load_config
from llm_build.data.hf_fetch import ensure_paths, resolve_train_files
from llm_build.data.memmap_dataset import MemmapTokenDataset
from llm_build.data.streaming_dataset import build_streaming_loader
from llm_build.model.lore_m3 import LoreM3
from llm_build.train.trainer import train_loop


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/experiment_100mb.yaml")
    args = p.parse_args()
    cfg = load_config(args.config)
    if cfg.data.streaming and cfg.data.train_files:
        ensure_paths(cfg.data.train_files, include_optional=False)
        cfg.data.train_files = resolve_train_files(cfg.data.train_files)
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
        meta_path = Path(cfg.paths.tokens_dir) / "meta.json"
        if meta_path.exists():
            import json

            with meta_path.open() as f:
                packed_seq = json.load(f).get("seq_len")
            if packed_seq is not None and packed_seq != cfg.train.seq_len:
                raise SystemExit(
                    f"Token bins were packed with seq_len={packed_seq} but config has "
                    f"train.seq_len={cfg.train.seq_len}. Re-run:\n"
                    f"  python scripts/preprocess_tokens.py --config {args.config}"
                )
        train_ds = MemmapTokenDataset(cfg.paths.tokens_dir, "train", seq_len=cfg.train.seq_len)
        val_ds = MemmapTokenDataset(cfg.paths.tokens_dir, "val", seq_len=cfg.train.seq_len)
        nw = cfg.train.num_workers
        use_cuda = cfg.train.device == "cuda" and torch.cuda.is_available()
        kw = {"num_workers": nw, "persistent_workers": nw > 0, "pin_memory": use_cuda}
        if nw > 0:
            kw["prefetch_factor"] = 2
        train_loader = DataLoader(
            train_ds, batch_size=cfg.train.micro_batch, shuffle=True, drop_last=True, **kw
        )
        val_loader = DataLoader(
            val_ds, batch_size=cfg.train.micro_batch, shuffle=False, drop_last=True, **kw
        )

    model = LoreM3(
        cfg.model,
        cfg.tokenizer.vocab_size,
        grad_checkpoint=cfg.train.grad_checkpoint,
    )
    try:
        import mlflow

        mlflow.set_experiment(cfg.train.mlflow_experiment)
        mlflow.start_run(run_name="lore-m3-train")
    except Exception as e:
        print(f"MLflow unavailable: {e}")
    train_loop(cfg, model, train_loader, val_loader)


if __name__ == "__main__":
    main()
