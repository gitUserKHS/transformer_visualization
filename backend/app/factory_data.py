import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

from .dataset import ROOT, StoryDataset
from .production_tokenizer import ProductionTokenizer


PREFERENCE_PATH = ROOT / "backend" / "data" / "user_preferences.jsonl"
EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
PHONE_PATTERN = re.compile(r"\b(?:\+?\d[\s-]?){8,15}\b")
WORD_PATTERN = re.compile(r"[A-Za-z']+")


@dataclass
class SFTExample:
    prompt: str
    response: str


@dataclass
class PreferenceExample:
    pair_id: str
    prompt: str
    chosen: str
    rejected: str
    source: str = "synthetic"


class FactoryData:
    def __init__(self, dataset: StoryDataset | None = None):
        self.dataset = dataset or StoryDataset()
        self.cleaned, self.cleaning_trace = self._clean(self.dataset.stories)
        self.tokenizer = ProductionTokenizer.load_or_train(self.cleaned, 8192)
        self.sft_examples = self._build_sft(1200)
        self.synthetic_preferences = self._build_preferences(1000)

    @staticmethod
    def _mask_pii(text: str) -> str:
        return PHONE_PATTERN.sub("[PHONE]", EMAIL_PATTERN.sub("[EMAIL]", text))

    @staticmethod
    def _quality_ok(text: str) -> bool:
        words = WORD_PATTERN.findall(text)
        return 20 <= len(words) <= 700 and len(set(word.lower() for word in words)) >= 10

    @staticmethod
    def _english_ok(text: str) -> bool:
        letters = sum(character.isascii() and character.isalpha() for character in text)
        alpha = sum(character.isalpha() for character in text)
        return alpha > 0 and letters / alpha >= 0.95

    @staticmethod
    def _repetition_ok(text: str) -> bool:
        words = [word.lower() for word in WORD_PATTERN.findall(text)]
        if len(words) < 12:
            return False
        trigrams = list(zip(words, words[1:], words[2:]))
        return len(set(trigrams)) / max(1, len(trigrams)) > 0.72

    def _clean(self, stories: list[str]) -> tuple[list[str], list[dict]]:
        current = list(stories)
        trace = [{"stage": "raw", "count": len(current)}]
        seen = set()
        unique = []
        for text in current:
            digest = hashlib.sha1(" ".join(text.lower().split()).encode()).hexdigest()
            if digest not in seen:
                seen.add(digest)
                unique.append(text)
        current = unique
        trace.append({"stage": "deduplicate", "count": len(current)})
        current = [text for text in current if self._quality_ok(text)]
        trace.append({"stage": "quality", "count": len(current)})
        current = [text for text in current if self._english_ok(text)]
        trace.append({"stage": "language", "count": len(current)})
        current = [text for text in current if self._repetition_ok(text)]
        trace.append({"stage": "repetition", "count": len(current)})
        current = [self._mask_pii(text) for text in current]
        trace.append({"stage": "pii_mask", "count": len(current)})
        return current, trace

    @staticmethod
    def _split_story(story: str) -> tuple[str, str]:
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", story.strip())
            if sentence.strip()
        ]
        if len(sentences) < 2:
            midpoint = max(1, len(story) // 2)
            return story[:midpoint], story[midpoint:]
        pivot = max(1, min(len(sentences) - 1, len(sentences) // 3))
        return " ".join(sentences[:pivot]), " ".join(sentences[pivot:])

    def _build_sft(self, count: int) -> list[SFTExample]:
        rng = random.Random(42)
        examples = []
        for story in rng.sample(self.cleaned, min(count, len(self.cleaned))):
            beginning, ending = self._split_story(story)
            examples.append(
                SFTExample(
                    prompt=f"Continue this children's story:\n{beginning}",
                    response=ending,
                )
            )
        while len(examples) < count:
            examples.extend(examples[: count - len(examples)])
        return examples[:count]

    @staticmethod
    def _corrupt(response: str, index: int) -> str:
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", response)
            if sentence.strip()
        ]
        mode = index % 4
        if mode == 0 and len(sentences) > 1:
            return " ".join(reversed(sentences))
        if mode == 1:
            words = response.split()
            fragment = " ".join(words[: min(12, len(words))])
            return " ".join([fragment] * 4)
        if mode == 2:
            return "Nothing made sense. Everyone was unhappy, and the story suddenly stopped."
        return response + " The answer is always always always the same because reward is reward."

    def _build_preferences(self, count: int) -> list[PreferenceExample]:
        preferences = []
        for index, example in enumerate(self.sft_examples[:count]):
            preferences.append(
                PreferenceExample(
                    pair_id=f"synthetic-{index:04d}",
                    prompt=example.prompt,
                    chosen=example.response,
                    rejected=self._corrupt(example.response, index),
                )
            )
        return preferences

    def append_vote(self, pair_id: str, choice: str) -> dict:
        existing = self.user_votes()
        if pair_id in existing:
            raise ValueError("이미 평가한 응답 쌍입니다.")
        pair = next(
            (item for item in self.synthetic_preferences if item.pair_id == pair_id),
            None,
        )
        if pair is None:
            raise KeyError(pair_id)
        swap = int(pair.pair_id.rsplit("-", 1)[-1]) % 2 == 1
        response_a = pair.rejected if swap else pair.chosen
        response_b = pair.chosen if swap else pair.rejected
        record = {
            "pair_id": pair_id,
            "choice": choice,
            "prompt": pair.prompt,
            "a": response_a,
            "b": response_b,
        }
        PREFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with PREFERENCE_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    @staticmethod
    def user_votes() -> dict[str, dict]:
        if not PREFERENCE_PATH.exists():
            return {}
        votes = {}
        for line in PREFERENCE_PATH.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                votes[record["pair_id"]] = record
        return votes

    def next_preference(self) -> dict:
        votes = self.user_votes()
        pair = next(
            (
                item
                for item in self.synthetic_preferences
                if item.pair_id not in votes
            ),
            self.synthetic_preferences[0],
        )
        swap = int(pair.pair_id.rsplit("-", 1)[-1]) % 2 == 1
        return {
            "pair_id": pair.pair_id,
            "prompt": pair.prompt,
            "response_a": pair.rejected if swap else pair.chosen,
            "response_b": pair.chosen if swap else pair.rejected,
            "revealed": False,
            "remaining": max(0, len(self.synthetic_preferences) - len(votes)),
        }

    def preference_training_data(self) -> list[PreferenceExample]:
        base = list(self.synthetic_preferences)
        user = []
        for record in self.user_votes().values():
            if record["choice"] == "tie":
                continue
            chosen_key = record["choice"]
            rejected_key = "b" if chosen_key == "a" else "a"
            user.append(
                PreferenceExample(
                    pair_id=f"user-{record['pair_id']}",
                    prompt=record["prompt"],
                    chosen=record[chosen_key],
                    rejected=record[rejected_key],
                    source="user",
                )
            )
        maximum_user = max(1, len(base) // 3)
        return base + user[-maximum_user:]

    def packed_tokens(self, context_length: int) -> list[list[int]]:
        stream = []
        for story in self.cleaned:
            stream.extend(self.tokenizer.encode(story))
        usable = len(stream) // (context_length + 1) * (context_length + 1)
        return [
            stream[index : index + context_length + 1]
            for index in range(0, usable, context_length + 1)
        ]

    def bootstrap(self) -> dict:
        raw = self.cleaning_trace[0]["count"]
        return {
            "cleaning_trace": [
                {
                    **item,
                    "retention": item["count"] / max(1, raw),
                }
                for item in self.cleaning_trace
            ],
            "sft_count": len(self.sft_examples),
            "preference_count": len(self.synthetic_preferences),
            "user_vote_count": len(self.user_votes()),
            "vocab_size": self.tokenizer.vocab_size,
        }
