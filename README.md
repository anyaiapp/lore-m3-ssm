# Lore-M3

A from-scratch **causal language model** for Apple Silicon (MPS) that mixes **Mamba-3-style state-space blocks** with **sparse grouped-query attention (GQA)**. Train on local chat JSONL, validate on a held-out set, then scale to multi-GB corpora with the same code path.

Hyperparameters for the main experiment live in **`configs/experiment_100mb.yaml`** only. Agents and automation should follow **[AGENTS.md](AGENTS.md)**.

---

## Architecture

**Model class:** `LoreM3` (`src/llm_build/model/lore_m3.py`)

| Component | Setting |
|-----------|---------|
| Depth | 20 layers |
| Layer schedule | `mamba3_5to1` — **GQA attention** at 1-based layers **5, 11, 17**; **Mamba-3 blocks** elsewhere |
| Width | `d_model=512`, SwiGLU MLP (`mlp_ratio=2.875`) |
| Attention | 8 query heads, 2 KV heads, `head_dim=64`, RoPE, QK-norm, causal **sliding window 512** |
| SSM | `d_state=128`, expand 2, post-gate RMSNorm; **PyTorch reference scan** (`ssm_backend: pytorch`, MPS-safe) |
| Embeddings | Tied input / LM head |
| Context | `max_seq_len=1024` (training `seq_len` matches config) |

**Tokenizer:** Hugging Face **byte-level BPE**, `vocab_size=32768`, specials `<|pad|>`, `<|user|>`, `<|assistant|>`, `<|eos|>`.

**Training objective:** next-token cross-entropy with labels masked on **user** tokens (assistant-only SFT). Chat format:

```text
<|user|>
{user text}
<|assistant|>
{assistant text}
```

**Inference:** `LoreM3.generate()` (full-sequence forward; optional wrapper in `src/llm_build/inference/generate.py`). KV/SSM caches are not implemented yet.

---

## Repository layout

```text
llm_build/
├── configs/
│   ├── experiment_100mb.yaml   # Main experiment (≤100 MB shard)
│   ├── smoke_train.yaml        # Tiny model/steps for CI and pipeline checks
│   └── scale_1gb.yaml          # Step 13: streaming multi-GB JSONL
├── data/jsonl/                 # Your raw JSONL (not committed if large)
├── scripts/
│   ├── make_experiment_shard.py
│   ├── train_tokenizer.py
│   ├── preprocess_tokens.py
│   ├── train.py
│   ├── eval.py
│   ├── chat.py
│   └── ui.py                    # Gradio studio (chat + metrics)
├── src/llm_build/
│   ├── config.py               # YAML loader + dataclasses
│   ├── data/                   # schema, chat template, encode, memmap, streaming
│   ├── model/                  # LoreM3, Mamba3, GQA transformer, RoPE, schedule
│   ├── train/                  # optimizer, scheduler, train loop
│   └── inference/generate.py
├── tests/                      # pytest (CPU smoke)
├── app.py                      # Delegates to scripts/chat.py
├── AGENTS.md                   # Step-by-step playbook for agents
└── artifacts/                  # Generated (gitignored): shards, tokenizer, bins, checkpoints
```

**Generated artifacts (typical experiment):**

| Path | Contents |
|------|----------|
| `artifacts/shards/experiment_100mb.jsonl` | Normalized chat JSONL (≤100 MB) |
| `artifacts/tokenizer/tokenizer.json` | Trained BPE |
| `artifacts/tokens/{train,val}.bin` | Packed `uint16` token ids |
| `artifacts/tokens/{train,val}.mask.bin` | Per-token assistant mask |
| `artifacts/tokens/meta.json` | `seq_len`, dtype, sequence counts |
| `artifacts/checkpoints/lore-m3/latest/` | `checkpoint.pt` + `meta.json` |

---

## Setup

```bash
cd llm_build
python -m venv .venv && source .venv/bin/activate
PIP_INDEX_URL=https://pypi.org/simple PIP_EXTRA_INDEX_URL= pip install -e ".[dev]"
```

If your global pip index points at a private mirror (e.g. CodeArtifact) and returns 401, keep the `PIP_INDEX_URL` override above.

Use **`.venv/bin/python3.14`** (or whichever interpreter installed the package) if `python` inside the venv is broken after a mixed uv install.

---

## How to test

**Unit / integration tests (CPU, no data shard required):**

```bash
pytest
# or
.venv/bin/python3.14 -m pytest
```

Covers norms, Mamba block backward, GQA block, `LoreM3` forward, layer schedule `[5,11,17]`, causal label masking, and a one-step train with a shrunk config.

**End-to-end pipeline smoke (~1 minute):**

Requires `data/jsonl/openhermes2_5.jsonl` (or an existing shard). Builds a small shard if you pass a lower byte cap:

```bash
.venv/bin/python3.14 scripts/make_experiment_shard.py --max-bytes 5000000
.venv/bin/python3.14 scripts/train_tokenizer.py --config configs/experiment_100mb.yaml
.venv/bin/python3.14 scripts/preprocess_tokens.py --config configs/experiment_100mb.yaml
MLFLOW_DISABLE_AGENT_HINT=1 .venv/bin/python3.14 scripts/train.py --config configs/smoke_train.yaml
.venv/bin/python3.14 scripts/eval.py --config configs/smoke_train.yaml \
  --checkpoint artifacts/checkpoints/lore-m3/latest
```

Use **`configs/smoke_train.yaml`** for eval/chat when the checkpoint was trained with that config (4 layers). Full checkpoints need **`configs/experiment_100mb.yaml`**.

---

## Full experiment (Steps 1–12)

The [AGENTS.md](AGENTS.md) playbook has **12 steps**. **Steps 1 and 5–10** are the repo itself (package layout, `LoreM3` model, trainer, configs)—you only run **install + pytest** once to verify them. **Steps 2–4** prepare data; **11** trains; **12** validates and demos inference. Run data and training commands **in order**. Do **not** put `data/jsonl/A_to_Z.jsonl.json` in the training shard (eval only).

| Step | What | You run |
|------|------|--------|
| **1** Scaffold | Project package, scripts, `configs/experiment_100mb.yaml`, tests | [Setup](#setup) + `pytest` |
| **2** Data | Normalized chat JSONL shard | `make_experiment_shard.py` |
| **3** Tokenizer | Byte BPE vocab on shard text | `train_tokenizer.py` |
| **4** Bins | Packed `uint16` tokens + assistant mask | `preprocess_tokens.py` |
| **5–9** Model | `LoreM3` (Mamba + GQA @ 5,11,17), RoPE, masked CE | *(code only)* |
| **10** Trainer | AdamW, AMP, checkpoints, MLflow hooks | *(code only)* |
| **11** Train run | SFT on memmap (or streaming in Step 13) | `train.py` |
| **12** Ship | Perplexity, A_to_Z holdout, chat / UI | `eval.py`, `chat.py`, `app.py` |

---

### Step 1 — Scaffold (install & verify)

**What:** Install `llm_build` in editable mode with dev deps (PyTorch, tokenizers, pytest, MLflow, etc.).

**Why:** Scripts import `llm_build.*`; tests catch broken installs and model shape bugs before you spend hours training.

**Gate:** `pytest` passes (CPU smoke tests, no 100 MB shard required).

---

### Step 2 — Build shard (≤100 MB)

**What:** `make_experiment_shard.py` reads raw JSONL, normalizes each line to `{"messages": [{"role":"user"|"assistant", "content":"..."}]}`, and writes one file until a **byte budget** is reached (default 100,000,000 bytes).

**Why:** OpenHermes is ~1.6 GB. A capped shard makes the **first** tokenizer + train loop fast and repeatable on one machine. The same scripts work later at scale (Step 13) with streaming.

**How:** Prepends smoke lines from `data/jsonl/llm_training_dataset.jsonl.json`, then streams `data/jsonl/openhermes2_5.jsonl` (ShareGPT / prompt-completion formats via `iter_records`).

**Output:** `artifacts/shards/experiment_100mb.jsonl`

```bash
.venv/bin/python3.14 scripts/make_experiment_shard.py --max-bytes 100000000
```

**Gate:** File exists; size ≤ `--max-bytes`; no A_to_Z lines inside.

---

### Step 3 — Train tokenizer

**What:** Trains a Hugging Face **byte-level BPE** tokenizer on chat-formatted text from the shard (`<|user|>` / `<|assistant|>` template), `vocab_size=32768`, plus specials `<|pad|>`, `<|user|>`, `<|assistant|>`, `<|eos|>`.

**Why:** Lore-M3 needs a fixed vocab and chat markers so training can mask loss on **assistant** tokens only (SFT, not full-sequence LM on user text).

**Output:** `artifacts/tokenizer/tokenizer.json`

```bash
.venv/bin/python3.14 scripts/train_tokenizer.py --config configs/experiment_100mb.yaml
```

**Gate:** Tokenizer loads; encode/decode round-trip on a sample user/assistant turn.

---

### Step 4 — Preprocess to memmap bins

**What:** Tokenizes every record in the shard, builds a flat token stream with a parallel **mask** (1 = train this token, 0 = user/padding), holds out `val_fraction` (default 5%) for validation, then **packs** into fixed-length chunks of `train.seq_len` (1024).

**Why:** Training reads memory-mapped `uint16` arrays—fast random access, no JSONL parsing in the hot loop. Masks implement assistant-only cross-entropy.

**Output:** `artifacts/tokens/train.bin`, `train.mask.bin`, `val.bin`, `val.mask.bin`, `meta.json`

```bash
.venv/bin/python3.14 scripts/preprocess_tokens.py --config configs/experiment_100mb.yaml
```

**Gate:** `meta.json` reports sequence counts; val bin non-empty.

---

### Steps 5–9 — Model (no command)

**What:** Single architecture **`LoreM3`**: 20 layers, `mamba3_5to1` (GQA attention at layers **5, 11, 17**), `d_model=512`, sliding-window attention 512, Mamba-3-style blocks with PyTorch reference scan on MPS.

**Why:** One hybrid design (SSM for throughput + sparse attention for quality); hyperparameters live only in `configs/experiment_100mb.yaml` so runs stay comparable.

**Gate:** `pytest` + optional forward on CPU/MPS with config-loaded shapes.

---

### Step 10 — Trainer (no command)

**What:** `train.py` + `Trainer`: AdamW, cosine schedule with warmup, gradient accumulation, grad clip, optional AMP, checkpointing, masked CE, MLflow metrics when available.

**Why:** Same loop for the 100 MB experiment and Step 13 scale (streaming loader swaps in when `data.streaming: true`).

---

### Step 11 — Train Lore-M3

**What:** Loads bins (or stream), builds full **20-layer** model from config, runs SFT for `max_steps` (default 20,000), saves `checkpoint.pt` + `meta.json` and `metrics.jsonl`.

**Why:** Produces the weights you evaluate and chat with. Effective batch = `micro_batch × grad_accum` (default 4 × 8 = 32 sequences per optimizer step).

**Output:** `artifacts/checkpoints/lore-m3/latest/` (and periodic `step_*` dirs)

```bash
MLFLOW_DISABLE_AGENT_HINT=1 .venv/bin/python3.14 scripts/train.py --config configs/experiment_100mb.yaml
```

Default training: `micro_batch=4`, `grad_accum=8`, AMP on MPS when available. Checkpoints every 500 steps.

**Performance note:** The reference Mamba scan loops over sequence length in Python; full 20×1024 steps on MPS are **slow**. Expect long wall-clock time; use `configs/smoke_train.yaml` for a quick pipeline check, or lower `max_steps` for a sanity run.

**Gate:** Loss decreases over first hundreds of steps; `latest/checkpoint.pt` loads without shape errors.

---

### Step 12 — Evaluate, chat, and UI

**What:** Measure perplexity on val memmap and on **A_to_Z** (never in shard), then interactive generation with the same chat template as training.

**Why:** Val bin tracks fit on in-distribution data; A_to_Z is a small, separate generalization check. Chat confirms the model + tokenizer + template work end-to-end.

**Val perplexity:**

```bash
.venv/bin/python3.14 scripts/eval.py --config configs/experiment_100mb.yaml \
  --checkpoint artifacts/checkpoints/lore-m3/latest
```

**A_to_Z holdout** (`data.eval_holdout` in config):

```bash
.venv/bin/python3.14 scripts/eval.py --config configs/experiment_100mb.yaml \
  --checkpoint artifacts/checkpoints/lore-m3/latest --holdout
```

**Terminal chat:**

```bash
.venv/bin/python3.14 scripts/chat.py --config configs/experiment_100mb.yaml \
  --checkpoint artifacts/checkpoints/lore-m3/latest
.venv/bin/python3.14 app.py --cli --config configs/experiment_100mb.yaml
```

**Gate:** Eval prints finite perplexity; chat returns coherent tokens (quality depends on training length).

#### Studio UI (optional)

**What:** Local Gradio app (`app.py` / `scripts/ui.py`) to load a checkpoint, chat, plot `metrics.jsonl`, and run quick health checks on val bins.

**Why:** Easier than the terminal for trying prompts and watching loss curves while a long Step 11 run is in progress.

Install the UI extra, then open the local web app:

```bash
PIP_INDEX_URL=https://pypi.org/simple pip install -e ".[ui]"
.venv/bin/python3.14 app.py
# or: .venv/bin/python3.14 scripts/ui.py --port 7860
```

| Tab | What it does |
|-----|----------------|
| **Chat** | Talk to the loaded checkpoint (temperature / max tokens) |
| **Metrics** | `metrics.jsonl` in the checkpoint dir (train / val loss, lr) |
| **Health & validation** | File checks, forward pass, quick val loss on `artifacts/tokens/val.*` |

After **Load model**, use the same `--config` as training (e.g. `smoke_train.yaml` vs `experiment_100mb.yaml`) so architecture matches the checkpoint.

Training writes `artifacts/checkpoints/lore-m3/metrics.jsonl` on each log step (and MLflow metrics when MLflow is available).

---

## Scale (Step 13)

**What:** Train on multi-GB JSONL via **streaming** (`data.streaming: true`, `data.train_files`) instead of one memmap shard.

**Why:** Reuse the tokenizer and trainer from the 100 MB experiment; avoid writing 1 GB+ shard files. Dolphin reasoning JSONL stays here—not in the experiment shard.

After the experiment tokenizer exists, train on large JSONL **without** building a single giant shard:

```bash
.venv/bin/python3.14 scripts/train.py --config configs/scale_1gb.yaml
```

`scale_1gb.yaml` sets `data.streaming: true` and reads `data.train_files` line-by-line. Val can still use `artifacts/tokens/val.*` from the experiment preprocess if present.

| File | Role |
|------|------|
| `data/jsonl/openhermes2_5.jsonl` | SFT stream |
| `data/jsonl/dolphin-r1-*.jsonl` | Reasoning / scale (multi-GB) |
| `data/jsonl/A_to_Z.jsonl.json` | **Eval only** — never in `train_files` |

---

## Data rules

- Training shard: OpenHermes + optional smoke JSONL only.
- **Never** train on `A_to_Z.jsonl.json`.
- JSONL supports ShareGPT `conversations` (`human`/`gpt`) and `prompt`/`completion` (including A_to_Z column names).

---

## MPS troubleshooting

| Issue | What to try |
|-------|-------------|
| NaN loss | Set `train.amp: false` in YAML or train Mamba scan in fp32 |
| OOM | Lower `micro_batch` or `seq_len` |
| Very slow steps | Expected with `ssm_backend: pytorch`; use `smoke_train.yaml` to verify the pipeline first |
| Wrong eval architecture | Match `--config` to the checkpoint (smoke vs `experiment_100mb`) |

---

## Config reference

| File | Purpose |
|------|---------|
| `configs/experiment_100mb.yaml` | Canonical model and training hyperparameters |
| `configs/smoke_train.yaml` | Small model, 16 steps, fast pipeline check |
| `configs/scale_1gb.yaml` | Larger `d_model`, streaming data, longer run |

All training scripts accept `--config path/to.yaml`.
