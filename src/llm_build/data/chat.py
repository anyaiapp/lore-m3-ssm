from __future__ import annotations

from llm_build.data.schema import Message, Record

USER_TAG = "<|user|>"
ASSISTANT_TAG = "<|assistant|>"
EOS_TAG = "<|eos|>"


def format_messages(messages: list[Message]) -> str:
    parts: list[str] = []
    for m in messages:
        if m.role == "user":
            parts.append(f"{USER_TAG}\n{m.content}\n")
        elif m.role == "assistant":
            parts.append(f"{ASSISTANT_TAG}\n{m.content}\n")
    return "".join(parts)


def format_record(rec: Record) -> str:
    return format_messages(rec.messages)
