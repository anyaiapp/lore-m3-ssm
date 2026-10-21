"""Fetch raw JSONL training files from Hugging Face when missing locally."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from llm_build.config import Config

_PKG_DIR = Path(__file__).resolve().parent
_BUNDLED_DIR = _PKG_DIR / "bundled"

# Fetched automatically by download_data / make_experiment_shard (no multi-GB extras).
DEFAULT_AUTO_FETCH: frozenset[str] = frozenset(
    {
        "data/jsonl/openhermes2_5.jsonl",
        "data/jsonl/llm_training_dataset.jsonl.json",
        "data/jsonl/A_to_Z.jsonl.json",
    }
)

# Download only when requested (--only or --optional on download_data.py).
OPTIONAL_DATA_PATHS: frozenset[str] = frozenset(
    {
        "data/jsonl/dolphin-r1-nonreasoning.jsonl",
        "data/jsonl/dolphin-r1-reasoning-deepseek.jsonl",
        "data/jsonl/dolphin-r1-reasoning-flash.jsonl",
    }
)


@dataclass(frozen=True)
class DataFileSpec:
    """How to obtain one file under data/jsonl/."""

    rel_path: str
    repo_id: str | None = None
    remote_filename: str | None = None
    bundled_name: str | None = None

    @property
    def local_path(self) -> Path:
        return Path(self.rel_path)


# Keys are paths relative to the repo root (as in configs).
DATA_FILE_SPECS: dict[str, DataFileSpec] = {
    "data/jsonl/openhermes2_5.jsonl": DataFileSpec(
        rel_path="data/jsonl/openhermes2_5.jsonl",
        repo_id="mathewhe/OpenHermes-2.5-Formatted",
        remote_filename="openhermes_2.5_teknium.jsonl",
    ),
    "data/jsonl/dolphin-r1-nonreasoning.jsonl": DataFileSpec(
        rel_path="data/jsonl/dolphin-r1-nonreasoning.jsonl",
        repo_id="cognitivecomputations/dolphin-r1",
        remote_filename="dolphin-r1-nonreasoning.jsonl",
    ),
    "data/jsonl/dolphin-r1-reasoning-deepseek.jsonl": DataFileSpec(
        rel_path="data/jsonl/dolphin-r1-reasoning-deepseek.jsonl",
        repo_id="cognitivecomputations/dolphin-r1",
        remote_filename="dolphin-r1-reasoning-deepseek.jsonl",
    ),
    "data/jsonl/dolphin-r1-reasoning-flash.jsonl": DataFileSpec(
        rel_path="data/jsonl/dolphin-r1-reasoning-flash.jsonl",
        repo_id="cognitivecomputations/dolphin-r1",
        remote_filename="dolphin-r1-reasoning-flash.jsonl",
    ),
    "data/jsonl/llm_training_dataset.jsonl.json": DataFileSpec(
        rel_path="data/jsonl/llm_training_dataset.jsonl.json",
        bundled_name="llm_training_dataset.jsonl.json",
    ),
    "data/jsonl/A_to_Z.jsonl.json": DataFileSpec(
        rel_path="data/jsonl/A_to_Z.jsonl.json",
        bundled_name="A_to_Z.jsonl.json",
    ),
}

# Common local typo: extra ".jsonl" suffix on the eval holdout file.
_PATH_ALIASES: dict[str, list[str]] = {
    "data/jsonl/A_to_Z.jsonl.json": ["data/jsonl/A_to_Z.jsonl.jsonl"],
}


def _is_present(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def resolve_data_path(path: str | Path) -> Path | None:
    """Return an existing non-empty file path, including known aliases."""
    p = Path(path)
    if _is_present(p):
        return p
    for alt in _PATH_ALIASES.get(str(path).replace("\\", "/"), []):
        ap = Path(alt)
        if _is_present(ap):
            return ap
    return None


def _copy_bundled(spec: DataFileSpec, dest: Path) -> None:
    if not spec.bundled_name:
        raise ValueError(f"No bundled copy for {spec.rel_path}")
    src = _BUNDLED_DIR / spec.bundled_name
    if not src.is_file():
        raise FileNotFoundError(f"Missing bundled data file: {src}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def _download_hf(spec: DataFileSpec, dest: Path) -> None:
    if not spec.repo_id or not spec.remote_filename:
        raise ValueError(f"No Hugging Face source for {spec.rel_path}")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError(
            "Install huggingface_hub to download datasets: pip install huggingface_hub"
        ) from e
    dest.parent.mkdir(parents=True, exist_ok=True)
    hf_hub_download(
        repo_id=spec.repo_id,
        repo_type="dataset",
        filename=spec.remote_filename,
        local_dir=str(dest.parent),
        local_dir_use_symlinks=False,
    )
    downloaded = dest.parent / spec.remote_filename
    if not downloaded.is_file():
        raise FileNotFoundError(f"Expected download at {downloaded}")
    if downloaded.resolve() != dest.resolve():
        if dest.exists():
            dest.unlink()
        shutil.move(str(downloaded), dest)


def ensure_jsonl_file(rel_path: str, *, force: bool = False) -> Path:
    """Download or copy ``rel_path`` if it is not already on disk."""
    key = rel_path.replace("\\", "/")
    spec = DATA_FILE_SPECS.get(key)
    if spec is None:
        dest = Path(rel_path)
        if _is_present(dest):
            return dest
        raise KeyError(
            f"Unknown data path {rel_path!r}. Add a DataFileSpec in hf_fetch.py or place the file manually."
        )
    existing = resolve_data_path(spec.local_path)
    if existing is not None and not force:
        return existing
    dest = spec.local_path
    if spec.bundled_name:
        _copy_bundled(spec, dest)
    else:
        _download_hf(spec, dest)
    return dest


def default_fetch_paths_for_config(cfg: Config) -> list[str]:
    """Paths to auto-download for a fresh clone (OpenHermes + small bundled files)."""
    paths: list[str] = []
    if cfg.data.source_for_shard:
        paths.append(cfg.data.source_for_shard)
    paths.extend(cfg.data.smoke_files)
    if cfg.data.eval_holdout:
        paths.append(cfg.data.eval_holdout)
    paths.append("data/jsonl/llm_training_dataset.jsonl.json")
    out: list[str] = []
    seen: set[str] = set()
    for p in paths:
        key = p.replace("\\", "/")
        if key in seen or key.startswith("artifacts/"):
            continue
        seen.add(key)
        if key in DEFAULT_AUTO_FETCH:
            out.append(key)
    return out


def resolve_train_files(files: list[str]) -> list[str]:
    """Return train JSONL paths that exist locally (optional files may be omitted)."""
    available: list[str] = []
    missing: list[str] = []
    for rel in files:
        key = rel.replace("\\", "/")
        found = resolve_data_path(key)
        if found is not None:
            available.append(str(found))
            continue
        p = Path(rel)
        if _is_present(p):
            available.append(str(p))
        else:
            missing.append(key)
    if not available:
        raise FileNotFoundError(
            "No training JSONL files found. Run:\n"
            "  python scripts/download_data.py --config configs/experiment_100mb.yaml"
        )
    if missing:
        optional = [m for m in missing if m in OPTIONAL_DATA_PATHS]
        other = [m for m in missing if m not in OPTIONAL_DATA_PATHS]
        if other:
            raise FileNotFoundError(f"Missing required training data: {', '.join(other)}")
        if optional:
            print(
                "Skipping optional data (not downloaded): "
                + ", ".join(optional)
                + "\n  python scripts/download_data.py --optional  # to fetch Dolphin R1"
            )
    return available


def ensure_paths(
    rels: list[str],
    *,
    force: bool = False,
    include_optional: bool = False,
) -> list[Path]:
    """Ensure each listed path exists when it has a known fetch spec."""
    resolved: list[Path] = []
    for rel in rels:
        key = rel.replace("\\", "/")
        if key in OPTIONAL_DATA_PATHS and not include_optional:
            existing = resolve_data_path(key)
            if existing is not None:
                resolved.append(existing)
            continue
        if key not in DATA_FILE_SPECS:
            p = Path(rel)
            if _is_present(p):
                resolved.append(p)
            continue
        resolved.append(ensure_jsonl_file(key, force=force))
    return resolved


def ensure_data_for_config(
    cfg: Config,
    *,
    force: bool = False,
    include_optional: bool = False,
) -> list[Path]:
    """Ensure default raw JSONL for ``cfg`` (OpenHermes + smoke/eval bundles)."""
    paths = list(default_fetch_paths_for_config(cfg))
    if include_optional:
        paths.extend(sorted(OPTIONAL_DATA_PATHS))
    return ensure_paths(paths, force=force, include_optional=include_optional)
