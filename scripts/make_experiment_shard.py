#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_build.data.chat import format_record
from llm_build.data.hf_fetch import ensure_jsonl_file, resolve_data_path
from llm_build.data.schema import iter_records


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--max-bytes", type=int, default=100_000_000)
    p.add_argument("--source", default="data/jsonl/openhermes2_5.jsonl")
    p.add_argument("--out", default="artifacts/shards/experiment_100mb.jsonl")
    p.add_argument("--smoke", nargs="*", default=["data/jsonl/llm_training_dataset.jsonl.json"])
    p.add_argument("--config", default="configs/experiment_100mb.yaml")
    p.add_argument("--skip-download", action="store_true", help="Do not fetch missing JSONL from Hugging Face")
    args = p.parse_args()
    if not args.skip_download:
        from llm_build.data.hf_fetch import DATA_FILE_SPECS

        def _maybe_fetch(rel: str) -> None:
            key = rel.replace("\\", "/")
            if key in DATA_FILE_SPECS:
                ensure_jsonl_file(key)

        _maybe_fetch(args.source)
        for smoke in args.smoke:
            _maybe_fetch(smoke)
    source = args.source
    resolved = resolve_data_path(source)
    if resolved is not None:
        source = str(resolved)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with out.open("w", encoding="utf-8") as wf:
        for smoke in args.smoke:
            sp = Path(smoke)
            if sp.exists():
                for rec in iter_records(str(sp)):
                    line = json.dumps(
                        {"messages": [{"role": m.role, "content": m.content} for m in rec.messages]},
                        ensure_ascii=False,
                    )
                    wf.write(line + "\n")
                    total += len(line.encode("utf-8"))
        for rec in iter_records(source, max_chars=16000):
            if total >= args.max_bytes:
                break
            line = json.dumps(
                {"messages": [{"role": m.role, "content": m.content} for m in rec.messages]},
                ensure_ascii=False,
            )
            b = len(line.encode("utf-8"))
            if total + b > args.max_bytes:
                break
            wf.write(line + "\n")
            total += b
    print(f"Wrote {out} ({total} bytes)")


if __name__ == "__main__":
    main()
