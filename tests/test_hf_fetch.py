from pathlib import Path

from llm_build.data.hf_fetch import (
    DATA_FILE_SPECS,
    OPTIONAL_DATA_PATHS,
    ensure_jsonl_file,
    ensure_paths,
    resolve_data_path,
    resolve_train_files,
)


def test_resolve_data_path_alias(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    root = Path("data/jsonl")
    root.mkdir(parents=True)
    alt = root / "A_to_Z.jsonl.jsonl"
    alt.write_text('{"prompt":"x","completion":"y"}\n', encoding="utf-8")
    resolved = resolve_data_path("data/jsonl/A_to_Z.jsonl.json")
    assert resolved == alt


def test_bundled_copy_skips_when_present(tmp_path: Path, monkeypatch) -> None:
    rel = "data/jsonl/llm_training_dataset.jsonl.json"
    dest = tmp_path / "data" / "jsonl" / "llm_training_dataset.jsonl.json"
    dest.parent.mkdir(parents=True)
    dest.write_text('{"prompt":"a","completion":"b"}\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    path = ensure_jsonl_file(rel)
    assert path.resolve() == dest.resolve()
    assert path.read_text(encoding="utf-8").startswith('{"prompt":"a"')


def test_bundled_copy_from_package(tmp_path: Path, monkeypatch) -> None:
    rel = "data/jsonl/A_to_Z.jsonl.json"
    monkeypatch.chdir(tmp_path)
    path = ensure_jsonl_file(rel)
    assert path.is_file()
    assert path.stat().st_size > 0
    assert "light-year" in path.read_text(encoding="utf-8")


def test_known_specs_cover_experiment_paths() -> None:
    keys = {
        "data/jsonl/openhermes2_5.jsonl",
        "data/jsonl/llm_training_dataset.jsonl.json",
        "data/jsonl/A_to_Z.jsonl.json",
    }
    assert keys <= set(DATA_FILE_SPECS)
    assert OPTIONAL_DATA_PATHS.isdisjoint(keys)


def test_ensure_paths_skips_optional_download(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    resolved = ensure_paths(list(OPTIONAL_DATA_PATHS), include_optional=False)
    assert resolved == []


def test_resolve_train_files_openhermes_only(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    root = Path("data/jsonl")
    root.mkdir(parents=True)
    oh = root / "openhermes2_5.jsonl"
    oh.write_text('{"conversations":[{"from":"human","value":"hi"},{"from":"gpt","value":"hey"}]}\n')
    files = resolve_train_files(
        [
            "data/jsonl/openhermes2_5.jsonl",
            "data/jsonl/dolphin-r1-nonreasoning.jsonl",
        ]
    )
    assert len(files) == 1
    assert Path(files[0]).resolve() == oh.resolve()
