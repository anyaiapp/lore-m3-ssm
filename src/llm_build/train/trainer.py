from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from llm_build.config import Config
from llm_build.model.lore_m3 import LoreM3
from llm_build.train.metrics import append_metric
from llm_build.train.optim import build_optimizer, build_scheduler
from llm_build.train.runtime import apply_model_runtime_opts


def resolve_device(name: str) -> torch.device:
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_checkpoint(
    path: Path,
    model: LoreM3,
    optimizer: torch.optim.Optimizer,
    step: int,
    cfg: Config,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg.raw,
        },
        path / "checkpoint.pt",
    )
    with (path / "meta.json").open("w") as f:
        json.dump({"step": step, "vocab_size": model.vocab_size}, f)


@torch.no_grad()
def run_validation(
    model: LoreM3,
    val_loader: DataLoader,
    device: torch.device,
    show_progress: bool = True,
) -> float:
    model.eval()
    total = 0.0
    n = 0
    batches = val_loader
    if show_progress:
        batches = tqdm(val_loader, desc="val", leave=False, unit="batch")
    for batch in batches:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        _, loss = model(input_ids, labels)
        total += loss.item()
        n += 1
    model.train()
    return total / max(n, 1)


def train_loop(
    cfg: Config,
    model: LoreM3,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
) -> None:
    device = resolve_device(cfg.train.device)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    model.to(device)
    for line in apply_model_runtime_opts(model, cfg, device):
        print(line)
    optimizer = build_optimizer(
        model,
        cfg.train.lr,
        cfg.train.weight_decay,
        fused=device.type == "cuda",
    )
    scheduler = build_scheduler(optimizer, cfg.train.warmup_steps, cfg.train.max_steps)
    use_amp = cfg.train.amp and device.type in ("cuda", "mps")
    ckpt_dir = Path(cfg.paths.checkpoint_dir)
    n_params = sum(p.numel() for p in model.parameters())
    batches_per_epoch = len(train_loader) if hasattr(train_loader, "__len__") else None
    eff_batch = cfg.train.micro_batch * cfg.train.grad_accum
    from llm_build.model.mamba3_ref import resolve_ssm_scan_mode

    ssm_scan = resolve_ssm_scan_mode(
        getattr(cfg.model, "ssm_scan", "auto"),
        cfg.train.seq_len,
        device,
    )
    print(
        f"Training on {device} | max_steps={cfg.train.max_steps} | "
        f"micro_batch={cfg.train.micro_batch} grad_accum={cfg.train.grad_accum} "
        f"(effective {eff_batch} seqs/step) | seq_len={cfg.train.seq_len} | "
        f"params={n_params/1e6:.1f}M | AMP={use_amp} | grad_ckpt={model.grad_checkpoint} | "
        f"ssm_scan={cfg.model.ssm_scan}→{ssm_scan}"
    )
    if batches_per_epoch is not None:
        print(f"Train loader: {batches_per_epoch} micro-batches per epoch")
    print(
        f"Each optimizer step = {cfg.train.grad_accum} micro-batches "
        f"(bar stays at 0% until all {cfg.train.grad_accum} finish)."
    )
    if device.type == "mps":
        print(
            "MPS: first micro-batch compiles Metal kernels; later micro-batches are faster. "
            "Use ssm_scan=auto (parallel prefix on MPS). Watch for 'micro k/N done' lines."
        )

    step = 0
    micro_step = 0
    model.train()
    optimizer.zero_grad(set_to_none=True)
    last_step_time = time.perf_counter()
    tokens_per_step = cfg.train.micro_batch * cfg.train.seq_len * cfg.train.grad_accum
    pbar = tqdm(total=cfg.train.max_steps, desc="train", unit="step", dynamic_ncols=True)
    while step < cfg.train.max_steps:
        for batch in train_loader:
            micro_in_step = (micro_step % cfg.train.grad_accum) + 1
            pbar.set_description(f"train (micro {micro_in_step}/{cfg.train.grad_accum})")
            t_micro = time.perf_counter()
            input_ids = batch["input_ids"].to(device, non_blocking=device.type != "mps")
            labels = batch["labels"].to(device, non_blocking=device.type != "mps")
            with torch.autocast(device_type=device.type, enabled=use_amp):
                _, loss = model(input_ids, labels)
                loss = loss / cfg.train.grad_accum
            loss.backward()
            micro_step += 1
            micro_secs = time.perf_counter() - t_micro
            tqdm.write(
                f"micro {micro_in_step}/{cfg.train.grad_accum} done in {micro_secs:.1f}s "
                f"(loss_chunk={loss.item() * cfg.train.grad_accum:.4f})"
            )
            accum_pos = micro_step % cfg.train.grad_accum
            if accum_pos != 0:
                pbar.set_postfix(accum=f"{accum_pos}/{cfg.train.grad_accum}", refresh=True)
            if micro_step % cfg.train.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                train_loss = loss.item() * cfg.train.grad_accum
                lr = scheduler.get_last_lr()[0]
                now = time.perf_counter()
                dt = max(now - last_step_time, 1e-9)
                tok_s = tokens_per_step / dt
                last_step_time = now

                if step % cfg.train.log_every == 0:
                    tqdm.write(f"step={step} loss={train_loss:.4f} lr={lr:.2e} tok/s≈{tok_s:,.0f}")
                    append_metric(
                        ckpt_dir,
                        {"step": step, "train_loss": train_loss, "val_loss": None, "lr": lr},
                    )
                    try:
                        import mlflow

                        mlflow.log_metric("train_loss", train_loss, step=step)
                        mlflow.log_metric("lr", lr, step=step)
                        mlflow.log_metric("tokens_per_sec", tok_s, step=step)
                    except Exception:
                        pass
                val_loss = None
                if val_loader is not None and step > 0 and step % cfg.train.save_every == 0:
                    val_loss = run_validation(model, val_loader, device)
                    tqdm.write(f"step={step} val_loss={val_loss:.4f}")
                    append_metric(
                        ckpt_dir,
                        {"step": step, "train_loss": None, "val_loss": val_loss, "lr": lr},
                    )
                    try:
                        import mlflow

                        mlflow.log_metric("val_loss", val_loss, step=step)
                    except Exception:
                        pass
                if step > 0 and step % cfg.train.save_every == 0:
                    tqdm.write(f"Saving checkpoint at step {step}...")
                    save_checkpoint(ckpt_dir / f"step_{step}", model, optimizer, step, cfg)
                    latest = ckpt_dir / "latest"
                    latest.mkdir(parents=True, exist_ok=True)
                    save_checkpoint(latest, model, optimizer, step, cfg)
                pbar.set_postfix(loss=f"{train_loss:.3f}", lr=f"{lr:.2e}", tok_s=f"{tok_s:,.0f}", refresh=False)
                pbar.update(1)
                step += 1
                if step >= cfg.train.max_steps:
                    break
    if micro_step % cfg.train.grad_accum != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        pbar.update(1)
    pbar.close()
    tqdm.write(f"Done. Final step={step}. Checkpoint: {ckpt_dir / 'latest'}")
    save_checkpoint(ckpt_dir / "latest", model, optimizer, step, cfg)
