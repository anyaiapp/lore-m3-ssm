#!/usr/bin/env python3
from __future__ import annotations

import argparse

from llm_build.config import load_config
from llm_build.data.hf_fetch import ensure_data_for_config, ensure_jsonl_file


def main() -> None:
    p = argparse.ArgumentParser(description="Download missing raw JSONL files from Hugging Face.")
    p.add_argument("--config", default="configs/experiment_100mb.yaml", help="YAML config (paths to ensure)")
    p.add_argument(
        "--only",
        action="append",
        metavar="PATH",
        help="Ensure a single path (e.g. data/jsonl/openhermes2_5.jsonl). Repeatable.",
    )
    p.add_argument(
        "--optional",
        action="store_true",
        help="Also download optional datasets (Dolphin R1 JSONL files)",
    )
    p.add_argument("--force", action="store_true", help="Re-download even if the file exists")
    args = p.parse_args()

    if args.only:
        for rel in args.only:
            path = ensure_jsonl_file(rel, force=args.force)
            print(f"OK {path}")
        return

    cfg = load_config(args.config)
    for path in ensure_data_for_config(
        cfg, force=args.force, include_optional=args.optional
    ):
        print(f"OK {path}")


if __name__ == "__main__":
    main()
