from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterator


@dataclass
class Message:
    role: str
    content: str


@dataclass
class Record:
    messages: list[Message]
    source: str = ""


def _pair_prompt_completion(obj: dict[str, Any]) -> Record | None:
    prompt = obj.get("prompt") or obj.get("Prompt (User)")
    completion = obj.get("completion") or obj.get("Completion (Assistant)")
    if not prompt or not completion:
        return None
    return Record(
        messages=[
            Message("user", str(prompt).strip()),
            Message("assistant", str(completion).strip()),
        ],
        source=str(obj.get("source", "prompt_completion")),
    )


def _pair_conversations(obj: dict[str, Any]) -> Record | None:
    convs = obj.get("conversations") or obj.get("messages")
    if not convs:
        return None
    messages: list[Message] = []
    for turn in convs:
        role = turn.get("from") or turn.get("role")
        content = turn.get("value") or turn.get("content")
        if not role or content is None:
            continue
        role = str(role).lower()
        if role in ("human", "user"):
            role = "user"
        elif role in ("gpt", "assistant", "bot"):
            role = "assistant"
        else:
            continue
        text = str(content).strip()
        if text:
            messages.append(Message(role, text))
    if len(messages) < 2:
        return None
    return Record(messages=messages, source=str(obj.get("source", "conversations")))


def parse_jsonl_line(line: str) -> Record | None:
    line = line.strip()
    if not line:
        return None
    obj = json.loads(line)
    rec = _pair_conversations(obj)
    if rec is not None:
        return rec
    return _pair_prompt_completion(obj)


def iter_records(path: str, max_chars: int | None = None) -> Iterator[Record]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = parse_jsonl_line(line)
            if rec is None:
                continue
            if max_chars is not None:
                total = sum(len(m.content) for m in rec.messages)
                if total > max_chars:
                    continue
            yield rec
