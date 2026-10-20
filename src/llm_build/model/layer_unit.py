from __future__ import annotations

import torch.nn as nn

from llm_build.model.mamba3_block import Mamba3Block
from llm_build.model.transformer import TransformerBlock


class LayerUnit(nn.Module):
    def __init__(self, layer_type: str, cfg):
        super().__init__()
        if layer_type == "attention":
            self.block = TransformerBlock(cfg)
        elif layer_type == "mamba3":
            self.block = Mamba3Block(cfg)
        else:
            raise ValueError(layer_type)

    def forward(self, x):
        return self.block(x)
