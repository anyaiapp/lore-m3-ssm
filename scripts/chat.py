#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from llm_build.config import load_config
from llm_build.data.chat import ASSISTANT_TAG, USER_TAG
from llm_build.data.tokenizer import TextTokenizer
from llm_build.model.lore_m3 import LoreM3
from llm_build.train.trainer import resolve_device


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/experiment_100mb.yaml")
    p.add_argument("--checkpoint", default="artifacts/checkpoints/lore-m3/latest")
    args = p.parse_args()
    cfg = load_config(args.config)
    device = resolve_device(cfg.train.device)
    tok = TextTokenizer(Path(cfg.paths.tokenizer_dir) / "tokenizer.json")
    model = LoreM3(cfg.model, cfg.tokenizer.vocab_size)
    ckpt = torch.load(Path(args.checkpoint) / "checkpoint.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    print("Lore-M3 chat (empty line to exit)")
    while True:
        user = input("You> ").strip()
        if not user:
            break
        prompt = f"{USER_TAG}\n{user}\n{ASSISTANT_TAG}\n"
        ids = torch.tensor([tok.encode(prompt)], device=device)
        out = model.generate(ids, max_new_tokens=128)
        text = tok.decode(out[0].tolist())
        if ASSISTANT_TAG in text:
            text = text.split(ASSISTANT_TAG)[-1]
        print(f"Assistant> {text.strip()}")


if __name__ == "__main__":
    main()
