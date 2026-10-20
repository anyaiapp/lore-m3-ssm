from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def metrics_path(checkpoint_dir: str | Path) -> Path:
    return Path(checkpoint_dir) / "metrics.jsonl"


def append_metric(checkpoint_dir: str | Path, record: dict[str, Any]) -> None:
    path = metrics_path(checkpoint_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"ts": time.time(), **record}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def load_metrics(checkpoint_dir: str | Path) -> list[dict[str, Any]]:
    path = metrics_path(checkpoint_dir)
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
