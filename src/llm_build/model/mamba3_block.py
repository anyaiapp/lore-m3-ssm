from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from llm_build.model.mamba3_ref import selective_scan
from llm_build.model.mlp import SwiGLU
from llm_build.model.norms import RMSNorm


class Mamba3Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.d_inner = cfg.d_model * cfg.expand
        self.norm1 = RMSNorm(cfg.d_model)
        self.in_proj = nn.Linear(cfg.d_model, self.d_inner * 2, bias=False)
        self.bc_proj = nn.Linear(self.d_inner, self.d_inner * 2, bias=False)
        self.bc_norm = RMSNorm(self.d_inner * 2)
        self.b_bias = nn.Parameter(torch.zeros(self.d_inner))
        self.c_bias = nn.Parameter(torch.zeros(self.d_inner))
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        self.a_log = nn.Parameter(torch.zeros(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, cfg.d_model, bias=False)
        self.post_gate_norm = RMSNorm(self.d_inner) if cfg.mamba_post_gate_norm else nn.Identity()
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg.d_model, cfg.mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        xz = self.in_proj(x)
        u, z = xz.chunk(2, dim=-1)
        u = F.silu(u)
        bc = self.bc_norm(self.bc_proj(u))
        b, c = bc.chunk(2, dim=-1)
        b = b + self.b_bias
        c = c + self.c_bias
        delta = self.dt_proj(u)
        a = self.a_log.unsqueeze(0).unsqueeze(0).expand_as(delta)
        y = selective_scan(u, delta, a, b, c, mode=self.cfg.ssm_scan, device=u.device)
        y = y * F.silu(z)
        y = self.post_gate_norm(y)
        x = residual + self.out_proj(y)
        x = x + self.mlp(self.norm2(x))
        return x
