import json
import random
from pathlib import Path

import torch

from .tokenizer import WordTokenizer


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "backend" / "data" / "tinystories.jsonl"
TOKENIZER_PATH = ROOT / "backend" / "artifacts" / "tokenizer.json"

FALLBACK_STORIES = [
    "Once upon a time, a little girl named Lily found a red ball. She shared it with her friend Tom, and they played all afternoon.",
    "Ben saw a small bird near the old tree. He gave it water, and the bird sang a happy song.",
    "Mia wanted to bake a cake for her mother. She mixed flour, milk, and sugar, then smiled when the cake was ready.",
    "A blue boat sailed across the quiet lake. Sam held the map while his sister watched the bright stars.",
    "The little dog was afraid of the rain. Ella sat beside him until the clouds moved away.",
    "Jack planted one tiny seed in the garden. Every day he gave it water, and soon a yellow flower appeared.",
    "Nora lost her green hat at the park. A kind boy found it under a bench and brought it back.",
    "Leo built a tall tower with wooden blocks. It fell down, so he laughed and tried again.",
]


class StoryDataset:
    def __init__(self, vocab_size: int = 2048):
        self.stories = self._load_stories()
        if TOKENIZER_PATH.exists():
            tokenizer = WordTokenizer.load(TOKENIZER_PATH)
            self.tokenizer = (
                tokenizer
                if len(tokenizer.vocab) == vocab_size
                else WordTokenizer.train(self.stories[:5000], vocab_size)
            )
        else:
            self.tokenizer = WordTokenizer.train(self.stories[:5000], vocab_size)
            self.tokenizer.save(TOKENIZER_PATH)
        split_index = min(5000, max(1, len(self.stories) - 1))
        train_stories = self.stories[:split_index]
        valid_stories = self.stories[split_index : split_index + 500] or self.stories[-2:]
        self.train_tokens = self._flatten(train_stories)
        self.valid_tokens = self._flatten(valid_stories)

    @staticmethod
    def _load_stories() -> list[str]:
        if DATA_PATH.exists():
            stories = []
            for line in DATA_PATH.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                text = row.get("text", "").strip()
                if text:
                    stories.append(text)
            if stories:
                return stories
        return [FALLBACK_STORIES[index % len(FALLBACK_STORIES)] for index in range(5500)]

    def _flatten(self, stories: list[str]) -> torch.Tensor:
        ids: list[int] = []
        for story in stories:
            ids.extend(self.tokenizer.encode(story))
        return torch.tensor(ids, dtype=torch.long)

    def batch(
        self,
        split: str,
        batch_size: int,
        context_length: int,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        data = self.train_tokens if split == "train" else self.valid_tokens
        max_start = len(data) - context_length - 1
        if max_start <= 0:
            raise RuntimeError("학습 데이터가 너무 짧습니다.")
        starts = torch.randint(
            0, max_start, (batch_size,), generator=generator
        ).tolist()
        x = torch.stack([data[start : start + context_length] for start in starts])
        y = torch.stack(
            [data[start + 1 : start + context_length + 1] for start in starts]
        )
        return x, y

    def sample_rows(self, count: int = 4) -> list[dict]:
        rng = random.Random(42)
        indices = rng.sample(range(min(5000, len(self.stories))), count)
        return [
            {
                "index": index,
                "text": self.stories[index],
                "tokens": self.tokenizer.split(self.stories[index])[:24],
            }
            for index in indices
        ]

    @property
    def is_official_subset(self) -> bool:
        return DATA_PATH.exists() and len(self.stories) >= 5500

