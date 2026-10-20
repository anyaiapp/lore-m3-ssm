from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from llm_build.config import ModelConfig
from llm_build.model.layer_schedule import format_layer_map, layer_types
from llm_build.model.layer_unit import LayerUnit
from llm_build.model.norms import RMSNorm


class LoreM3(nn.Module):
    def __init__(self, cfg: ModelConfig, vocab_size: int):
        super().__init__()
        self.cfg = cfg
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, cfg.d_model)
        types = layer_types(cfg.n_layers, cfg.layer_schedule)
        self.layers = nn.ModuleList([LayerUnit(t, cfg) for t in types])
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight
        self._init_weights()
        print(f"LoreM3 layer map: {format_layer_map(cfg.n_layers, cfg.layer_schedule)}")

    def _init_weights(self):
        std = 0.02
        nn.init.normal_(self.embed.weight, mean=0.0, std=std)
        if not self.cfg.tie_embeddings:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=std)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        logits = self.lm_head(self.norm(x))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            ids = input_ids[:, -self.cfg.max_seq_len :]
            logits, _ = self.forward(ids)
            next_logits = logits[:, -1, :] / max(temperature, 1e-5)
            probs = F.softmax(next_logits, dim=-1)
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cum = torch.cumsum(sorted_probs, dim=-1)
            mask = cum > top_p
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = False
            filtered = sorted_probs.masked_fill(mask, 0.0)
            filtered = filtered / filtered.sum(dim=-1, keepdim=True)
            next_id = sorted_idx.gather(-1, torch.multinomial(filtered, 1))
            input_ids = torch.cat([input_ids, next_id], dim=-1)
        return input_ids
