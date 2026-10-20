from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer


class TextTokenizer:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._tok = Tokenizer.from_file(str(self.path))

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids, skip_special_tokens=False)

    def token_to_id(self, token: str) -> int:
        return self._tok.token_to_id(token)
