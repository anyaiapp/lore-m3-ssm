# Lore-M3 — agent playbook

Execute **Steps 1→12** in order from the plan. Pass each verify gate before continuing.

## Rules

1. One model: `LoreM3`, schedule `mamba3_5to1`, attention at layers 5, 11, 17 (L=20).
2. Config: `configs/experiment_100mb.yaml` only for hyperparameters.
3. Never train on `data/jsonl/A_to_Z.jsonl.json`.
4. Dolphin multi-GB JSONL = Step 13 only.

## Commands

```bash
source .venv/bin/activate
PIP_INDEX_URL=https://pypi.org/simple PIP_EXTRA_INDEX_URL= pip install -e ".[dev]"
# Prefer `.venv/bin/python3.14` if `python` in the venv is broken (uv mixed installs).
python scripts/make_experiment_shard.py --max-bytes 100000000
python scripts/train_tokenizer.py --config configs/experiment_100mb.yaml
python scripts/preprocess_tokens.py --config configs/experiment_100mb.yaml
python scripts/train.py --config configs/experiment_100mb.yaml
python scripts/eval.py --checkpoint artifacts/checkpoints/lore-m3/latest
python scripts/chat.py --checkpoint artifacts/checkpoints/lore-m3/latest
python app.py   # Gradio UI (pip install -e ".[ui]" first)
```
