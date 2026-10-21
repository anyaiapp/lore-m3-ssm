"""Device-specific runtime optimizations (compile, backends)."""

from __future__ import annotations

import torch

from llm_build.config import Config
from llm_build.model.lore_m3 import LoreM3
from llm_build.model.mamba3_block import Mamba3Block
from llm_build.model.transformer import TransformerBlock


def resolve_ssm_backend(name: str, device: torch.device) -> str:
    if name == "auto":
        return "compiled" if device.type == "cuda" else "pytorch"
    return name


def apply_model_runtime_opts(model: LoreM3, cfg: Config, device: torch.device) -> list[str]:
    """Mutate model in-place for faster training on the target device. Returns log lines."""
    lines: list[str] = []
    backend = resolve_ssm_backend(cfg.model.ssm_backend, device)
    if backend == "compiled" and device.type == "cuda" and hasattr(torch, "compile"):
        n_mamba = 0
        n_attn = 0
        for layer in model.layers:
            if isinstance(layer.block, Mamba3Block):
                layer.block = torch.compile(layer.block, mode="reduce-overhead")
                n_mamba += 1
            elif isinstance(layer.block, TransformerBlock):
                layer.block = torch.compile(layer.block, mode="reduce-overhead")
                n_attn += 1
        lines.append(
            f"torch.compile: {n_mamba} Mamba3 + {n_attn} attention blocks (CUDA, reduce-overhead)"
        )
    elif backend == "compiled" and device.type != "cuda":
        lines.append(f"ssm_backend={cfg.model.ssm_backend}→pytorch (compile skipped on {device.type})")
    else:
        lines.append(f"ssm_backend={backend} (PyTorch reference scan)")
    return lines
