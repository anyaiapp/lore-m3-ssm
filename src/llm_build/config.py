from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    name: str = "lore_m3"
    n_layers: int = 20
    layer_schedule: str = "mamba3_5to1"
    d_model: int = 512
    n_heads: int = 8
    n_kv_heads: int = 2
    head_dim: int = 64
    d_state: int = 128
    expand: int = 2
    mlp_ratio: float = 2.875
    ssm: str = "mamba3"
    ssm_variant: str = "siso"
    ssm_backend: str = "pytorch"  # pytorch | auto | compiled (CUDA: torch.compile blocks)
    ssm_scan: str = "auto"  # auto | triton | compiled | parallel | sequential
    mamba_post_gate_norm: bool = True
    attention_pos: str = "rope"
    attn_window: int | None = 512
    qk_norm: bool = True
    tie_embeddings: bool = True
    max_seq_len: int = 1024
    mimo_rank: int = 4


@dataclass
class TokenizerConfig:
    backend: str = "hf"
    vocab_size: int = 32768


@dataclass
class TrainConfig:
    micro_batch: int = 4
    seq_len: int = 1024
    grad_accum: int = 8
    lr: float = 3e-4
    warmup_steps: int = 200
    max_steps: int = 20000
    weight_decay: float = 0.1
    device: str = "mps"
    amp: bool = True
    grad_checkpoint: bool = True
    seed: int = 42
    val_fraction: float = 0.05
    log_every: int = 50
    save_every: int = 500
    mlflow_experiment: str = "lore-m3-experiment-100mb"
    num_workers: int = 0  # memmap: 0; streaming on Linux can use 2–4


@dataclass
class DataConfig:
    shard: str | None = "artifacts/shards/experiment_100mb.jsonl"
    smoke_files: list[str] = field(default_factory=list)
    source_for_shard: str = "data/jsonl/openhermes2_5.jsonl"
    eval_holdout: str = "data/jsonl/A_to_Z.jsonl.json"
    max_chars_per_example: int = 16000
    train_files: list[str] = field(default_factory=list)
    streaming: bool = False


@dataclass
class PathsConfig:
    tokenizer_dir: str = "artifacts/tokenizer"
    tokens_dir: str = "artifacts/tokens"
    checkpoint_dir: str = "artifacts/checkpoints/lore-m3"


@dataclass
class Config:
    model: ModelConfig
    tokenizer: TokenizerConfig
    train: TrainConfig
    data: DataConfig
    paths: PathsConfig
    raw: dict[str, Any] = field(default_factory=dict)


def _merge_dict(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if k == "_base_":
            continue
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge_dict(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path) -> Config:
    path = Path(path)
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    if "_base_" in data:
        base_path = path.parent / data["_base_"]
        with base_path.open() as f:
            base = yaml.safe_load(f) or {}
        data = _merge_dict(base, data)
    data.pop("_base_", None)
    return Config(
        model=ModelConfig(**data.get("model", {})),
        tokenizer=TokenizerConfig(**data.get("tokenizer", {})),
        train=TrainConfig(**data.get("train", {})),
        data=DataConfig(**data.get("data", {})),
        paths=PathsConfig(**data.get("paths", {})),
        raw=data,
    )
