import json
import re
from collections import Counter
from pathlib import Path


TOKEN_PATTERN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[^\w\s]", re.UNICODE)
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


class WordTokenizer:
    def __init__(self, vocab: list[str]):
        if vocab[:4] != SPECIAL_TOKENS:
            raise ValueError("어휘의 첫 네 항목은 특수 토큰이어야 합니다.")
        self.vocab = vocab
        self.token_to_id = {token: index for index, token in enumerate(vocab)}

    @classmethod
    def train(cls, texts: list[str], vocab_size: int = 2048) -> "WordTokenizer":
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(cls.split(text))
        ordered = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
        vocab = SPECIAL_TOKENS + [token for token, _ in ordered[: vocab_size - 4]]
        while len(vocab) < vocab_size:
            vocab.append(f"<unused_{len(vocab)}>")
        return cls(vocab)

    @staticmethod
    def split(text: str) -> list[str]:
        return TOKEN_PATTERN.findall(text.lower())

    def encode(
        self, text: str, add_bos: bool = True, add_eos: bool = True
    ) -> list[int]:
        ids = [self.token_to_id.get(token, self.unk_id) for token in self.split(text)]
        return ([self.bos_id] if add_bos else []) + ids + (
            [self.eos_id] if add_eos else []
        )

    def decode(self, ids: list[int]) -> str:
        tokens = [
            self.vocab[index]
            for index in ids
            if 0 <= index < len(self.vocab)
            and self.vocab[index] not in {"<pad>", "<bos>", "<eos>"}
        ]
        text = " ".join(tokens)
        text = re.sub(r"\s+([.,!?;:])", r"\1", text)
        text = re.sub(r"([(\"'])\s+", r"\1", text)
        return text

    def token_strings(self, ids: list[int]) -> list[str]:
        return [self.vocab[index] if 0 <= index < len(self.vocab) else "<unk>" for index in ids]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"vocab": self.vocab}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "WordTokenizer":
        return cls(json.loads(path.read_text(encoding="utf-8"))["vocab"])

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def unk_id(self) -> int:
        return 1

    @property
    def bos_id(self) -> int:
        return 2

    @property
    def eos_id(self) -> int:
        return 3

