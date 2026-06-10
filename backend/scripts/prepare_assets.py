"""Download a fixed TinyStories subset and train the bundled demo checkpoint."""

import argparse
import json
import random
import sys
from pathlib import Path

import httpx
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.app.dataset import DATA_PATH, TOKENIZER_PATH, StoryDataset
from backend.app.engine import DEMO_PATH
from backend.app.model import VisualGPT
from backend.app.schemas import ModelConfig


ROWS_URL = "https://datasets-server.huggingface.co/rows"


def download_subset(count: int = 5500) -> None:
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows: list[str] = []
    offsets = list(range(0, count, 100))
    random.Random(42).shuffle(offsets)
    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        for number, offset in enumerate(offsets, start=1):
            response = client.get(
                ROWS_URL,
                params={
                    "dataset": "roneneldan/TinyStories",
                    "config": "default",
                    "split": "train",
                    "offset": offset,
                    "length": min(100, count - len(rows)),
                },
            )
            response.raise_for_status()
            for row in response.json().get("rows", []):
                text = row.get("row", {}).get("text", "").strip()
                if text:
                    rows.append(text)
            print(f"dataset {number}/{len(offsets)}: {len(rows)} stories")
    if len(rows) < count:
        raise RuntimeError(f"Expected {count} stories, received {len(rows)}")
    with DATA_PATH.open("w", encoding="utf-8") as handle:
        for text in rows[:count]:
            handle.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    TOKENIZER_PATH.unlink(missing_ok=True)


def train_demo(steps: int = 600) -> None:
    torch.manual_seed(42)
    dataset = StoryDataset()
    config = ModelConfig()
    model = VisualGPT(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    generator = torch.Generator().manual_seed(42)
    model.train()
    for step in range(1, steps + 1):
        x, y = dataset.batch("train", 32, config.context_length, generator)
        _, loss, _ = model(x, y)
        assert loss is not None
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 50 == 0:
            print(f"checkpoint {step}/{steps}: loss={float(loss.detach()):.4f}")
    DEMO_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": config.model_dump()}, DEMO_PATH)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--steps", type=int, default=600)
    args = parser.parse_args()
    if not args.skip_download:
        download_subset()
    train_demo(args.steps)


if __name__ == "__main__":
    main()
