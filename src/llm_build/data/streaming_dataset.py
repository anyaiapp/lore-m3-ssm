from __future__ import annotations

import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, IterableDataset

from llm_build.config import Config
from llm_build.data.encode import batch_to_causal_labels, encode_record_with_mask
from llm_build.data.schema import iter_records
from llm_build.data.tokenizer import TextTokenizer


class StreamingTokenDataset(IterableDataset):
    """Stream JSONL files into fixed-length training chunks (no full-file load)."""

    def __init__(
        self,
        files: list[str],
        tokenizer_path: str | Path,
        seq_len: int,
        max_chars_per_example: int = 16000,
        seed: int = 42,
    ):
        self.files = [str(f) for f in files]
        self.tokenizer_path = Path(tokenizer_path)
        self.seq_len = seq_len
        self.max_chars = max_chars_per_example
        self.seed = seed

    def _record_stream(self):
        paths = list(self.files)
        rng = random.Random(self.seed)
        while True:
            rng.shuffle(paths)
            for path in paths:
                for rec in iter_records(path, self.max_chars):
                    yield rec

    def __iter__(self):
        tok = TextTokenizer(self.tokenizer_path / "tokenizer.json")
        buf_ids: list[int] = []
        buf_mask: list[int] = []
        stream = self._record_stream()
        while True:
            while len(buf_ids) < self.seq_len:
                rec = next(stream)
                ids, mask = encode_record_with_mask(tok, rec)
                buf_ids.extend(ids)
                buf_mask.extend(mask)
            chunk_ids = buf_ids[: self.seq_len]
            chunk_mask = buf_mask[: self.seq_len]
            buf_ids = buf_ids[self.seq_len :]
            buf_mask = buf_mask[self.seq_len :]
            input_ids = torch.tensor(chunk_ids, dtype=torch.long)
            mask = torch.tensor(chunk_mask, dtype=torch.long)
            labels = batch_to_causal_labels(input_ids.unsqueeze(0), mask.unsqueeze(0)).squeeze(0)
            yield {"input_ids": input_ids, "labels": labels}


def build_streaming_loader(cfg: Config) -> DataLoader:
    files = cfg.data.train_files
    if not files:
        raise ValueError("data.train_files required when data.streaming is true")
    ds = StreamingTokenDataset(
        files=files,
        tokenizer_path=cfg.paths.tokenizer_dir,
        seq_len=cfg.train.seq_len,
        max_chars_per_example=cfg.data.max_chars_per_example,
        seed=cfg.train.seed,
    )
    workers = getattr(cfg.train, "num_workers", 0)
    return DataLoader(
        ds,
        batch_size=cfg.train.micro_batch,
        num_workers=workers,
        prefetch_factor=2 if workers > 0 else None,
        persistent_workers=workers > 0,
    )
