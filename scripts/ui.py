#!/usr/bin/env python3
"""Gradio studio: chat, metrics, and health checks after training."""

from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr

from llm_build.inference.session import LoreSession

_SESSION = LoreSession()


def _load(config_path: str, checkpoint_dir: str) -> tuple[str, str, str, list[list]]:
    status = _SESSION.load(config_path, checkpoint_dir)
    health = _SESSION.health_report()
    metrics_md = _SESSION.metrics_summary()
    table = _SESSION.metrics_dataframe_rows()
    return status, health, metrics_md, table


def _chat(message: str, history: list, max_tokens: int, temperature: float) -> tuple[list, str]:
    if not message.strip():
        return history or [], ""
    history = list(history or [])
    reply = _SESSION.chat(message, max_new_tokens=max_tokens, temperature=temperature)
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": reply})
    return history, ""


def _refresh_metrics() -> tuple[str, list[list]]:
    return _SESSION.metrics_summary(), _SESSION.metrics_dataframe_rows()


def _run_val() -> str:
    return _SESSION.quick_validation()


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Lore-M3 Studio") as demo:
        gr.Markdown(
            "# Lore-M3 Studio\n"
            "Load a trained checkpoint, chat with the model, and inspect loss / validation."
        )
        with gr.Row():
            config_path = gr.Textbox(
                label="Config YAML",
                value="configs/smoke_train.yaml",
                scale=2,
            )
            checkpoint_dir = gr.Textbox(
                label="Checkpoint directory",
                value="artifacts/checkpoints/lore-m3-smoke/latest",
                scale=2,
            )
            load_btn = gr.Button("Load model", variant="primary", scale=1)

        status = gr.Markdown("Click **Load model** after training.")

        with gr.Tabs():
            with gr.Tab("Chat"):
                chatbot = gr.Chatbot(label="Conversation", height=420)
                with gr.Row():
                    user_in = gr.Textbox(
                        label="Your message",
                        placeholder="Ask something…",
                        scale=4,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)
                with gr.Accordion("Generation", open=False):
                    max_tokens = gr.Slider(16, 512, value=128, step=8, label="Max new tokens")
                    temperature = gr.Slider(0.1, 1.5, value=0.8, step=0.05, label="Temperature")

            with gr.Tab("Metrics"):
                metrics_md = gr.Markdown("Load a checkpoint to see training metrics.")
                metrics_table = gr.Dataframe(
                    headers=["step", "train_loss", "val_loss", "lr"],
                    label="metrics.jsonl",
                    interactive=False,
                )
                refresh_btn = gr.Button("Refresh metrics")

            with gr.Tab("Health & validation"):
                health_md = gr.Markdown("")
                val_btn = gr.Button("Run validation (quick)", variant="secondary")
                val_out = gr.Markdown("")

        load_btn.click(
            _load,
            inputs=[config_path, checkpoint_dir],
            outputs=[status, health_md, metrics_md, metrics_table],
        )
        send_btn.click(
            _chat,
            inputs=[user_in, chatbot, max_tokens, temperature],
            outputs=[chatbot, user_in],
        )
        user_in.submit(
            _chat,
            inputs=[user_in, chatbot, max_tokens, temperature],
            outputs=[chatbot, user_in],
        )
        refresh_btn.click(_refresh_metrics, outputs=[metrics_md, metrics_table])
        val_btn.click(_run_val, outputs=[val_out])

    return demo


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true", help="Create a public Gradio link")
    args = p.parse_args()
    demo = build_demo()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
