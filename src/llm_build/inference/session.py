from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from llm_build.config import Config, ModelConfig, load_config
from llm_build.data.chat import ASSISTANT_TAG, USER_TAG
from llm_build.data.memmap_dataset import MemmapTokenDataset
from llm_build.data.tokenizer import TextTokenizer
from llm_build.model.lore_m3 import LoreM3
from llm_build.model.layer_schedule import format_layer_map
from llm_build.train.metrics import load_metrics
from llm_build.train.trainer import resolve_device, run_validation


def model_config_from_dict(data: dict) -> ModelConfig:
    allowed = {f.name for f in fields(ModelConfig)}
    return ModelConfig(**{k: v for k, v in data.items() if k in allowed})


class LoreSession:
    """Loaded checkpoint + tokenizer for chat and eval in the UI."""

    def __init__(self) -> None:
        self.cfg: Config | None = None
        self.model: LoreM3 | None = None
        self.tokenizer: TextTokenizer | None = None
        self.device: torch.device | None = None
        self.step: int = 0
        self.checkpoint_dir: Path | None = None
        self.load_error: str | None = None

    @property
    def ready(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def load(self, config_path: str, checkpoint_dir: str) -> str:
        self.load_error = None
        try:
            self.cfg = load_config(config_path)
            ckpt_root = Path(checkpoint_dir)
            ckpt_file = ckpt_root / "checkpoint.pt"
            if not ckpt_file.is_file():
                raise FileNotFoundError(f"No checkpoint at {ckpt_file}")

            tok_path = Path(self.cfg.paths.tokenizer_dir) / "tokenizer.json"
            if not tok_path.is_file():
                raise FileNotFoundError(f"Tokenizer missing: {tok_path}")

            self.device = resolve_device(self.cfg.train.device)
            payload = torch.load(ckpt_file, map_location=self.device, weights_only=False)
            model_cfg = self.cfg.model
            if isinstance(payload.get("config"), dict) and "model" in payload["config"]:
                model_cfg = model_config_from_dict(payload["config"]["model"])

            vocab = self.cfg.tokenizer.vocab_size
            self.model = LoreM3(model_cfg, vocab)
            self.model.load_state_dict(payload["model"])
            self.model.to(self.device)
            self.model.eval()

            self.tokenizer = TextTokenizer(tok_path)
            self.step = int(payload.get("step", 0))
            self.checkpoint_dir = ckpt_root
            return self.status_markdown()
        except Exception as e:
            self.model = None
            self.tokenizer = None
            self.load_error = str(e)
            return f"**Load failed:** {e}"

    def status_markdown(self) -> str:
        if not self.ready or self.cfg is None:
            return self.load_error or "Model not loaded."
        mcfg = self.model.cfg if self.model is not None else self.cfg.model
        layer_map = format_layer_map(mcfg.n_layers, mcfg.layer_schedule)
        metrics = load_metrics(self.checkpoint_dir or Path("."))
        last_loss = next(
            (r for r in reversed(metrics) if r.get("train_loss") is not None),
            None,
        )
        loss_line = ""
        if last_loss:
            loss_line = f"\n- Last logged train loss (step {last_loss.get('step')}): **{last_loss['train_loss']:.4f}**"
        tok_dir = Path(self.cfg.paths.tokenizer_dir)
        tokens_dir = Path(self.cfg.paths.tokens_dir)
        return (
            f"### Model ready\n"
            f"- Checkpoint: `{self.checkpoint_dir}` (step **{self.step}**)\n"
            f"- Device: **{self.device}**\n"
            f"- Layers: **{mcfg.n_layers}** — `{layer_map}`\n"
            f"- `d_model={mcfg.d_model}`, vocab **{self.cfg.tokenizer.vocab_size}**\n"
            f"- Tokenizer: `{'ok' if (tok_dir / 'tokenizer.json').is_file() else 'missing'}`\n"
            f"- Val bins: `{'ok' if (tokens_dir / 'val.bin').is_file() else 'missing'}`"
            f"{loss_line}"
        )

    def chat(self, user_message: str, max_new_tokens: int = 128, temperature: float = 0.8) -> str:
        if not self.ready or self.model is None or self.tokenizer is None or self.device is None:
            return "Load a checkpoint first."
        user_message = (user_message or "").strip()
        if not user_message:
            return ""
        prompt = f"{USER_TAG}\n{user_message}\n{ASSISTANT_TAG}\n"
        ids = torch.tensor([self.tokenizer.encode(prompt)], device=self.device)
        out = self.model.generate(
            ids,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
        )
        text = self.tokenizer.decode(out[0].tolist())
        if ASSISTANT_TAG in text:
            text = text.split(ASSISTANT_TAG)[-1]
        return text.strip()

    def quick_validation(self, max_batches: int = 8) -> str:
        if not self.ready or self.model is None or self.cfg is None or self.device is None:
            return "Load a checkpoint first."
        val_bin = Path(self.cfg.paths.tokens_dir) / "val.bin"
        if not val_bin.is_file():
            return "No `artifacts/tokens/val.bin` — run preprocess_tokens.py first."
        ds = MemmapTokenDataset(self.cfg.paths.tokens_dir, "val", seq_len=self.cfg.train.seq_len)
        loader = DataLoader(ds, batch_size=min(self.cfg.train.micro_batch, 4))
        total = 0.0
        n = 0
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                if n >= max_batches:
                    break
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                _, loss = self.model(input_ids, labels)
                total += loss.item()
                n += 1
        avg = total / max(n, 1)
        ppl = float(torch.exp(torch.tensor(avg)))
        return f"Val loss (≈{n} batches): **{avg:.4f}** · perplexity ≈ **{ppl:.2f}**"

    def health_report(self) -> str:
        if self.cfg is None:
            return "Load config / checkpoint to run health checks."
        lines = ["### Health checks"]
        ckpt = self.checkpoint_dir / "checkpoint.pt" if self.checkpoint_dir else None
        lines.append(f"- Checkpoint file: **{'pass' if ckpt and ckpt.is_file() else 'fail'}**")
        tok = Path(self.cfg.paths.tokenizer_dir) / "tokenizer.json"
        lines.append(f"- Tokenizer: **{'pass' if tok.is_file() else 'fail'}**")
        meta = Path(self.cfg.paths.tokens_dir) / "meta.json"
        lines.append(f"- Token meta: **{'pass' if meta.is_file() else 'fail'}**")
        if self.ready and self.model is not None and self.device is not None:
            try:
                x = torch.randint(0, min(512, self.cfg.tokenizer.vocab_size), (1, 8), device=self.device)
                logits, _ = self.model(x)
                ok = logits.shape[-1] == self.cfg.tokenizer.vocab_size
                lines.append(f"- Forward pass: **{'pass' if ok else 'fail'}**")
            except Exception as e:
                lines.append(f"- Forward pass: **fail** ({e})")
        else:
            lines.append("- Forward pass: *skipped (model not loaded)*")
        return "\n".join(lines)

    def metrics_dataframe_rows(self) -> list[list]:
        if not self.checkpoint_dir:
            return []
        rows = load_metrics(self.checkpoint_dir)
        out: list[list] = []
        for r in rows:
            out.append(
                [
                    r.get("step"),
                    r.get("train_loss"),
                    r.get("val_loss"),
                    r.get("lr"),
                ]
            )
        return out

    def metrics_summary(self) -> str:
        rows = load_metrics(self.checkpoint_dir) if self.checkpoint_dir else []
        if not rows:
            return (
                "No `metrics.jsonl` in the checkpoint directory yet. "
                "Re-run training with the current code to record loss curves, "
                "or use **Run validation** for a live loss estimate."
            )
        train_pts = [r for r in rows if r.get("train_loss") is not None]
        val_pts = [r for r in rows if r.get("val_loss") is not None]
        text = f"**{len(train_pts)}** train log points, **{len(val_pts)}** val points.\n\n"
        if train_pts:
            first, last = train_pts[0], train_pts[-1]
            text += (
                f"Train loss: {first['train_loss']:.4f} (step {first.get('step')}) → "
                f"{last['train_loss']:.4f} (step {last.get('step')})\n"
            )
        if val_pts:
            v = val_pts[-1]
            text += f"Latest val loss: **{v['val_loss']:.4f}** (step {v.get('step')})\n"
        return text
