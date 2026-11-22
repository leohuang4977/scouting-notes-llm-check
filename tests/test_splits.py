"""The dev and test splits must not overlap, and scoring must use test notes only."""

import pandas as pd

from src.code_keywords import code_notes
from src.codebook import DATA_DIR, N_DEV, N_NOTES, note_id, read_notes, read_sheet, split_of
from src.score import score


def test_split_rule():
    assert split_of("N0001") == "dev"
    assert split_of(note_id(N_DEV)) == "dev"
    assert split_of(note_id(N_DEV + 1)) == "test"
    assert split_of(note_id(N_NOTES)) == "test"


def test_committed_splits_do_not_overlap_and_cover_every_note():
    notes = read_notes()
    dev = set(notes.loc[notes["split"] == "dev", "note_id"])
    test = set(notes.loc[notes["split"] == "test", "note_id"])
    assert dev.isdisjoint(test)
    assert dev | test == set(notes["note_id"])
    assert (len(dev), len(test)) == (150, 450)
    assert (notes["split"] == notes["note_id"].map(split_of)).all()


def test_scoring_uses_only_test_notes():
    notes = read_notes()
    human = read_sheet(DATA_DIR / "human_codes.csv")
    keyword = code_notes(notes)
    table = score(notes, human, {"keyword": keyword}, n_boot=10)["agreement_by_field"]
    assert set(table.loc[table["notes"] == "all test notes", "n"]) == {450}
    held_out = table.loc[table["notes"] == "held-out wording", "n"]
    assert (held_out > 0).all() and (held_out < 450).all()


def test_dropping_dev_notes_does_not_change_the_scores():
    notes = read_notes()
    human = read_sheet(DATA_DIR / "human_codes.csv")
    keyword = code_notes(notes)
    full = score(notes, human, {"keyword": keyword}, n_boot=10)["agreement_by_field"]
    test_only = notes[notes["split"] == "test"]
    reduced = score(test_only, human, {"keyword": keyword}, n_boot=10)["agreement_by_field"]
    pd.testing.assert_frame_equal(full, reduced)
