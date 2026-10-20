from __future__ import annotations

import torch

from llm_build.data.chat import ASSISTANT_TAG, USER_TAG
from llm_build.data.schema import Record
from llm_build.data.tokenizer import TextTokenizer


def encode_record_with_mask(tok: TextTokenizer, rec: Record) -> tuple[list[int], list[int]]:
    ids: list[int] = []
    mask: list[int] = []
    for msg in rec.messages:
        prefix = f"{USER_TAG}\n" if msg.role == "user" else f"{ASSISTANT_TAG}\n"
        chunk = prefix + msg.content + "\n"
        piece_ids = tok.encode(chunk)
        ids.extend(piece_ids)
        mask.extend([0] * len(piece_ids) if msg.role == "user" else [1] * len(piece_ids))
    return ids, mask


def batch_to_causal_labels(input_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Next-token labels; -100 where we should not train (user tokens / last pos)."""
    labels = input_ids.clone()
    labels[:, :-1] = input_ids[:, 1:]
    labels[:, -1] = -100
    train_mask = torch.zeros_like(labels, dtype=torch.bool)
    train_mask[:, :-1] = mask[:, 1:] > 0
    return labels.masked_fill(~train_mask, -100)
