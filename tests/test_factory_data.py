import json

from backend.app import factory_data
from backend.app.factory_data import FactoryData, PreferenceExample


def test_votes_respect_hidden_ab_order_and_are_capped_at_25_percent(
    tmp_path, monkeypatch
):
    preference_path = tmp_path / "preferences.jsonl"
    monkeypatch.setattr(factory_data, "PREFERENCE_PATH", preference_path)
    data = FactoryData.__new__(FactoryData)
    data.synthetic_preferences = [
        PreferenceExample(
            pair_id=f"synthetic-{index:04d}",
            prompt=f"prompt {index}",
            chosen=f"chosen {index}",
            rejected=f"rejected {index}",
        )
        for index in range(12)
    ]

    odd_pair = data.next_preference()
    assert odd_pair["pair_id"] == "synthetic-0000"
    data.append_vote("synthetic-0000", "a")
    data.append_vote("synthetic-0001", "b")
    for index in range(2, 10):
        data.append_vote(
            f"synthetic-{index:04d}", "a" if index % 2 == 0 else "b"
        )

    records = [
        json.loads(line)
        for line in preference_path.read_text(encoding="utf-8").splitlines()
    ]
    assert records[1]["b"] == "chosen 1"
    training = data.preference_training_data()
    user_rows = [row for row in training if row.source == "user"]
    assert all(row.chosen.startswith("chosen ") for row in user_rows)
    assert len(user_rows) == 4
    assert len(user_rows) / len(training) == 0.25


def test_duplicate_vote_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(factory_data, "PREFERENCE_PATH", tmp_path / "votes.jsonl")
    data = FactoryData.__new__(FactoryData)
    data.synthetic_preferences = [
        PreferenceExample("synthetic-0000", "prompt", "chosen", "rejected")
    ]
    data.append_vote("synthetic-0000", "a")

    try:
        data.append_vote("synthetic-0000", "b")
    except ValueError:
        pass
    else:
        raise AssertionError("중복 투표가 허용되었습니다.")
