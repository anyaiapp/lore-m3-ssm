from __future__ import annotations


def attention_layer_indices(n_layers: int, schedule: str) -> list[int]:
    """1-based layer indices that use attention."""
    if schedule != "mamba3_5to1":
        raise ValueError(f"Unsupported schedule: {schedule}")
    indices: list[int] = []
    # Evenly spaced attention layers for L=20 -> 5, 11, 17
    stride = 6
    i = 5
    while i <= n_layers:
        indices.append(i)
        i += stride
    return indices


def layer_types(n_layers: int, schedule: str) -> list[str]:
    attn = set(attention_layer_indices(n_layers, schedule))
    types: list[str] = []
    for layer in range(1, n_layers + 1):
        types.append("attention" if layer in attn else "mamba3")
    return types


def format_layer_map(n_layers: int, schedule: str) -> str:
    types = layer_types(n_layers, schedule)
    parts = [f"L{i+1:02d}:{t[:1].upper()}" for i, t in enumerate(types)]
    return ", ".join(parts)
