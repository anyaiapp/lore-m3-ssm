#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from llm_build.config import load_config
from llm_build.data.encode import encode_record_with_mask
from llm_build.data.schema import iter_records
from llm_build.data.tokenizer import TextTokenizer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/experiment_100mb.yaml")
    args = p.parse_args()
    cfg = load_config(args.config)
    tok = TextTokenizer(Path(cfg.paths.tokenizer_dir) / "tokenizer.json")
    assert cfg.data.shard is not None
    records = list(iter_records(cfg.data.shard, cfg.data.max_chars_per_example))
    n_val = max(1, int(len(records) * cfg.train.val_fraction))
    val_recs = records[:n_val]
    train_recs = records[n_val:]

    def pack(split_recs, split_name: str):
        all_ids: list[int] = []
        all_mask: list[int] = []
        for rec in split_recs:
            ids, mask = encode_record_with_mask(tok, rec)
            all_ids.extend(ids)
            all_mask.extend(mask)
        seq_len = cfg.train.seq_len
        n_seq = len(all_ids) // seq_len
        all_ids = all_ids[: n_seq * seq_len]
        all_mask = all_mask[: n_seq * seq_len]
        out_dir = Path(cfg.paths.tokens_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        dtype = np.uint16 if cfg.tokenizer.vocab_size < 65536 else np.uint32
        np.asarray(all_ids, dtype=dtype).tofile(out_dir / f"{split_name}.bin")
        np.asarray(all_mask, dtype=np.uint8).tofile(out_dir / f"{split_name}.mask.bin")
        return n_seq, dtype

    train_n, dtype = pack(train_recs, "train")
    val_n, _ = pack(val_recs, "val")
    meta = {
        "seq_len": cfg.train.seq_len,
        "dtype": np.dtype(dtype).name,
        "train_seqs": train_n,
        "val_seqs": val_n,
        "vocab_size": cfg.tokenizer.vocab_size,
    }
    with (Path(cfg.paths.tokens_dir) / "meta.json").open("w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
