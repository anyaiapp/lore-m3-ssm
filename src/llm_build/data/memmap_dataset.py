from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from llm_build.data.encode import batch_to_causal_labels


class MemmapTokenDataset(Dataset):
    def __init__(self, tokens_dir: str | Path, split: str = "train", seq_len: int | None = None):
        self.dir = Path(tokens_dir)
        meta_path = self.dir / "meta.json"
        with meta_path.open() as f:
            meta = json.load(f)
        self.packed_len = int(meta["seq_len"])
        self.seq_len = min(seq_len or self.packed_len, self.packed_len)
        self.dtype = np.dtype(meta.get("dtype", "uint16"))
        tokens = np.memmap(self.dir / f"{split}.bin", dtype=self.dtype, mode="r")
        mask = np.memmap(self.dir / f"{split}.mask.bin", dtype=np.uint8, mode="r")
        n_seq = len(tokens) // self.packed_len
        self.tokens = tokens
        self.mask = mask
        self.n_seq = n_seq

    def __len__(self) -> int:
        return self.n_seq

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start = idx * self.packed_len
        end = start + self.seq_len
        x = np.asarray(self.tokens[start:end], dtype=np.int64)
        m = np.asarray(self.mask[start:end], dtype=np.int64)
        input_ids = torch.from_numpy(x)
        mask = torch.from_numpy(m)
        labels = batch_to_causal_labels(input_ids.unsqueeze(0), mask.unsqueeze(0)).squeeze(0)
        return {"input_ids": input_ids, "labels": labels}
