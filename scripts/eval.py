#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from llm_build.config import load_config
from llm_build.data.encode import batch_to_causal_labels, encode_record_with_mask
from llm_build.data.memmap_dataset import MemmapTokenDataset
from llm_build.data.hf_fetch import ensure_jsonl_file, resolve_data_path
from llm_build.data.schema import iter_records
from llm_build.data.tokenizer import TextTokenizer
from llm_build.model.lore_m3 import LoreM3
from llm_build.train.trainer import resolve_device


class HoldoutJsonlDataset(Dataset):
    def __init__(self, path: str, tokenizer_path: Path, seq_len: int, max_examples: int = 256):
        self.seq_len = seq_len
        self.tok = TextTokenizer(tokenizer_path / "tokenizer.json")
        self.chunks: list[tuple[torch.Tensor, torch.Tensor]] = []
        buf_ids: list[int] = []
        buf_mask: list[int] = []
        n = 0
        for rec in iter_records(path):
            if n >= max_examples:
                break
            ids, mask = encode_record_with_mask(self.tok, rec)
            buf_ids.extend(ids)
            buf_mask.extend(mask)
            while len(buf_ids) >= seq_len:
                chunk_ids = buf_ids[:seq_len]
                chunk_mask = buf_mask[:seq_len]
                buf_ids = buf_ids[seq_len:]
                buf_mask = buf_mask[seq_len:]
                input_ids = torch.tensor(chunk_ids, dtype=torch.long)
                mask = torch.tensor(chunk_mask, dtype=torch.long)
                labels = batch_to_causal_labels(input_ids.unsqueeze(0), mask.unsqueeze(0)).squeeze(0)
                self.chunks.append((input_ids, labels))
            n += 1

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        input_ids, labels = self.chunks[idx]
        return {"input_ids": input_ids, "labels": labels}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/experiment_100mb.yaml")
    p.add_argument("--checkpoint", default="artifacts/checkpoints/lore-m3/latest")
    p.add_argument("--holdout", action="store_true", help="Eval on data.eval_holdout (A_to_Z)")
    args = p.parse_args()
    cfg = load_config(args.config)
    device = resolve_device(cfg.train.device)
    model = LoreM3(cfg.model, cfg.tokenizer.vocab_size)
    ckpt = torch.load(Path(args.checkpoint) / "checkpoint.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    if args.holdout:
        holdout = cfg.data.eval_holdout
        if resolve_data_path(holdout) is None:
            ensure_jsonl_file(holdout)
        holdout_path = resolve_data_path(holdout) or holdout
        ds = HoldoutJsonlDataset(
            str(holdout_path),
            Path(cfg.paths.tokenizer_dir),
            cfg.train.seq_len,
        )
        print(f"holdout {cfg.data.eval_holdout}: {len(ds)} chunks")
    else:
        ds = MemmapTokenDataset(cfg.paths.tokens_dir, "val")
    loader = DataLoader(ds, batch_size=cfg.train.micro_batch)
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            _, loss = model(input_ids, labels)
            total_loss += loss.item()
            n += 1
    avg = total_loss / max(n, 1)
    ppl = torch.exp(torch.tensor(avg))
    print(f"loss={avg:.4f} ppl≈{ppl.item():.2f}")


if __name__ == "__main__":
    main()
