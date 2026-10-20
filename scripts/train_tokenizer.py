#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from llm_build.config import load_config
from llm_build.data.chat import format_record
from llm_build.data.schema import iter_records


def text_iterator(paths: list[str], max_chars: int | None):
    for path in paths:
        for rec in iter_records(path, max_chars=max_chars):
            yield format_record(rec)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/experiment_100mb.yaml")
    args = p.parse_args()
    cfg = load_config(args.config)
    out_dir = Path(cfg.paths.tokenizer_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    specials = ["<|pad|>", "<|user|>", "<|assistant|>", "<|eos|>"]
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=cfg.tokenizer.vocab_size,
        special_tokens=specials,
        show_progress=True,
    )
    if cfg.data.shard:
        paths = [cfg.data.shard]
    elif cfg.data.train_files:
        paths = cfg.data.train_files
    else:
        raise SystemExit("Set data.shard or data.train_files in config")
    tokenizer.train_from_iterator(
        text_iterator(paths, cfg.data.max_chars_per_example),
        trainer=trainer,
    )
    path = out_dir / "tokenizer.json"
    tokenizer.save(str(path))
    print(f"Saved tokenizer to {path} (vocab={tokenizer.get_vocab_size()})")


if __name__ == "__main__":
    main()
