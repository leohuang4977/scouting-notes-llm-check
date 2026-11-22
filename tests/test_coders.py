"""Basic checks on the simulated human coders and the keyword list."""

from collections import Counter

from src.code_keywords import code_notes
from src.codebook import DATA_DIR, HUMAN_CODERS, load_codebook, read_notes, read_sheet
from src.simulate_coders import simulate


def test_committed_human_codes_match_the_simulation():
    expected = simulate(read_notes()).to_csv(index=False, lineterminator="\n")
    committed = (DATA_DIR / "human_codes.csv").read_text(encoding="utf-8").replace("\r\n", "\n")
    assert committed == expected


def test_consensus_is_the_majority_whenever_there_is_one():
    human = read_sheet(DATA_DIR / "human_codes.csv")
    for field in load_codebook():
        wide = human.pivot(index="note_id", columns="coder", values=field.name)
        for _, row in wide.iterrows():
            top, count = Counter(row[c] for c in HUMAN_CODERS).most_common(1)[0]
            if count >= 2:
                assert row["consensus"] == top


def test_every_code_is_an_allowed_value():
    notes = read_notes()
    sheets = [read_sheet(DATA_DIR / "human_codes.csv"), code_notes(notes)]
    for sheet in sheets:
        for field in load_codebook():
            assert set(sheet[field.name]) <= set(field.values), field.name


def test_shared_habit_makes_the_consensus_wrong_on_hedged_health_mentions():
    """The planted habit: correlated misses on hedged mentions, so the consensus is often wrong there."""
    notes = read_notes().set_index("note_id")
    human = read_sheet(DATA_DIR / "human_codes.csv")
    wide = human.pivot(index="note_id", columns="coder", values="health_flag").loc[notes.index]
    hedged = notes["hedged_health"]
    plain_yes = (notes["true_health_flag"] == "yes") & ~hedged & (notes["hard_fields"] == "")
    assert (wide.loc[hedged, "consensus"] == "yes").mean() < 0.8
    assert (wide.loc[plain_yes, "consensus"] == "yes").mean() > 0.95
    missed = wide.loc[hedged, HUMAN_CODERS] == "no"
    all_three = missed.all(axis=1).mean()
    if_independent = missed.mean().prod()
    assert all_three > if_independent  # misses come together, not one at a time
