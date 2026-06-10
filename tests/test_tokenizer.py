from backend.app.tokenizer import SPECIAL_TOKENS, WordTokenizer


def test_tokenizer_is_deterministic_and_keeps_special_ids():
    texts = ["A red ball, a blue ball.", "The blue bird."]
    first = WordTokenizer.train(texts, vocab_size=32)
    second = WordTokenizer.train(list(reversed(texts)), vocab_size=32)
    assert first.vocab == second.vocab
    assert first.vocab[:4] == SPECIAL_TOKENS
    assert first.encode("A blue bird.") == second.encode("A blue bird.")


def test_tokenizer_round_trip_is_readable():
    tokenizer = WordTokenizer.train(["Once upon a time, there was a cat."], 32)
    decoded = tokenizer.decode(tokenizer.encode("Once upon a time, there was a cat."))
    assert decoded == "once upon a time, there was a cat."

