"""Text generation helpers (full-sequence forward; KV/SSM cache optional later)."""

from __future__ import annotations

import torch

from llm_build.model.lore_m3 import LoreM3


@torch.no_grad()
def generate(
    model: LoreM3,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 0.8,
    top_p: float = 0.9,
) -> torch.Tensor:
    return model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
