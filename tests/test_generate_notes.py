"""The note generator must be deterministic and must match the committed data."""

import re

import pandas as pd

from src import codebook
from src.codebook import DATA_DIR, load_codebook
from src.generate_notes import generate_notes


def test_same_seed_gives_same_notes():
    first = generate_notes(n=60)
    second = generate_notes(n=60)
    pd.testing.assert_frame_equal(first, second)


def test_a_different_seed_gives_different_notes(monkeypatch):
    baseline = generate_notes(n=60)
    monkeypatch.setattr(codebook, "SEED", 1)
    other = generate_notes(n=60)
    assert not baseline["text"].equals(other["text"])


def test_committed_notes_match_the_generator():
    expected = generate_notes().to_csv(index=False, lineterminator="\n")
    committed = (DATA_DIR / "notes.csv").read_text(encoding="utf-8").replace("\r\n", "\n")
    assert committed == expected


def test_notes_have_the_shape_the_spec_asks_for():
    notes = generate_notes()
    assert len(notes) == 600
    counted = notes["text"].map(lambda t: len(re.split(r"(?<=[.!?])\s+", t.strip())))
    assert (counted == notes["n_sentences"]).all()
    assert notes["n_sentences"].between(2, 5).all()
    assert 0.10 <= notes["ambiguous"].mean() <= 0.20
    assert 0.02 <= notes["has_typo"].mean() <= 0.08
    for field in load_codebook():
        assert set(notes[f"true_{field.name}"]) <= set(field.values), field.name


def test_held_out_wording_never_appears_in_dev_notes():
    """Every sentence in a _new list must be absent from dev notes and flagged when used in test."""
    import re as regex

    from src import phrases as P

    def new_sentences(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key.endswith("_new"):
                    yield from (value.values() if isinstance(value, dict) else value)
                else:
                    yield from new_sentences(value)

    notes = generate_notes()
    dev_text = " ".join(notes.loc[notes["split"] == "dev", "text"])
    banks = [P.OPENERS, P.VELO, P.POWER, P.COMMAND, P.APPROACH, P.MAKEUP, P.HEALTH, P.PROJECTION, P.CLOSING]
    for bank in banks:
        for sentence in new_sentences(bank):
            fixed_part = max(regex.split(r"\{[^}]*\}", sentence), key=len).strip(" ,.")
            assert fixed_part not in dev_text, sentence
    assert (notes.loc[notes["split"] == "dev", "heldout_fields"] == "").all()
    assert (notes.loc[notes["split"] == "test", "heldout_fields"] != "").mean() > 0.5
