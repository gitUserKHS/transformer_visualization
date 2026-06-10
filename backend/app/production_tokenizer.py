from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from .dataset import ROOT


TOKENIZER_PATH = ROOT / "backend" / "artifacts" / "production_tokenizer.json"
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>", "<user>", "<assistant>"]


class ProductionTokenizer:
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    @classmethod
    def load_or_train(
        cls, texts: list[str], vocab_size: int = 8192
    ) -> "ProductionTokenizer":
        if TOKENIZER_PATH.exists():
            return cls(Tokenizer.from_file(str(TOKENIZER_PATH)))
        tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tokenizer.decoder = ByteLevelDecoder()
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=2,
            special_tokens=SPECIAL_TOKENS,
            show_progress=False,
        )
        tokenizer.train_from_iterator(texts, trainer=trainer)
        TOKENIZER_PATH.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(TOKENIZER_PATH))
        return cls(tokenizer)

    def encode(
        self, text: str, add_bos: bool = True, add_eos: bool = True
    ) -> list[int]:
        ids = self.tokenizer.encode(text).ids
        return ([self.bos_id] if add_bos else []) + ids + (
            [self.eos_id] if add_eos else []
        )

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def token_strings(self, ids: list[int]) -> list[str]:
        return [self.tokenizer.id_to_token(token_id) or "<unk>" for token_id in ids]

    def id(self, token: str) -> int:
        value = self.tokenizer.token_to_id(token)
        if value is None:
            raise KeyError(token)
        return value

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    @property
    def pad_id(self) -> int:
        return self.id("<pad>")

    @property
    def unk_id(self) -> int:
        return self.id("<unk>")

    @property
    def bos_id(self) -> int:
        return self.id("<bos>")

    @property
    def eos_id(self) -> int:
        return self.id("<eos>")

    @property
    def user_id(self) -> int:
        return self.id("<user>")

    @property
    def assistant_id(self) -> int:
        return self.id("<assistant>")

