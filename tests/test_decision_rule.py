"""The decision rule in score.decide, checked on small hand-made tables."""

import pandas as pd

from src.score import SUBSETS, decide

EACH, PAIRS = "each human coder (mean)", "human coder pairs (mean)"


def table(ceiling, model, keyword_heldout, model_heldout_kappa, ceiling_heldout=0.85, model_heldout_each=None):
    """Rows for one field. model and keyword_heldout are (kappa, ac1, kappa_with_each_coder)."""
    rows = []

    def add(coder, reference, notes, kappa, ac1=1.0):
        rows.append({"field": "role", "coder": coder, "reference": reference, "notes": notes,
                     "kappa": kappa, "ac1": ac1})

    a, h = SUBSETS["all"], SUBSETS["held_out"]
    add("human_ceiling", PAIRS, a, ceiling)
    add("human_ceiling", PAIRS, h, ceiling_heldout)
    add("m", "consensus", a, model[0], model[1])
    add("m", EACH, a, model[2])
    add("m", "consensus", h, model_heldout_kappa)
    add("m", EACH, h, model[2] if model_heldout_each is None else model_heldout_each)
    add("keyword", "consensus", a, 0.9)
    add("keyword", "consensus", h, keyword_heldout[0], keyword_heldout[1])
    add("keyword", EACH, h, keyword_heldout[2])
    return pd.DataFrame(rows)


def verdict(df):
    return decide(df, ["m"])["verdict"].iloc[0]


def test_a_field_where_people_disagree_needs_codebook_work_first():
    assert verdict(table(0.55, (0.9, 0.9, 0.9), (0.9, 0.9, 0.9), 0.9)) == "codebook needs work"


def test_ceiling_clause_is_one_sided_and_like_for_like():
    # Mean kappa with each coder .76 is .09 below a .85 ceiling: passes.
    assert verdict(table(0.85, (0.8, 0.8, 0.76), (0.2, 0.2, 0.2), 0.8)) == "model"
    # .74 is .11 below: fails, even though kappa with the consensus is high.
    assert verdict(table(0.85, (0.9, 0.9, 0.74), (0.2, 0.2, 0.2), 0.9)) == "human coders"
    # Being above the ceiling never fails the clause.
    assert verdict(table(0.70, (0.9, 0.9, 0.95), (0.2, 0.2, 0.2), 0.9)) == "model"


def test_keyword_preference_is_decided_on_held_out_wording():
    # Keyword passes on held-out wording and the model is not .05 better there: keyword list.
    assert verdict(table(0.85, (0.9, 0.9, 0.85), (0.8, 0.8, 0.8), 0.84)) == "keyword list"
    # The model is .05 better on held-out wording: model.
    assert verdict(table(0.85, (0.9, 0.9, 0.85), (0.8, 0.8, 0.8), 0.85)) == "model"
    # Keyword fails on held-out wording: it is not preferred, however it does overall.
    assert verdict(table(0.85, (0.9, 0.9, 0.85), (0.5, 0.5, 0.5), 0.7)) == "model"


def test_model_must_also_meet_the_bar_on_held_out_wording():
    # Strong on all test notes, but kappa .60 with the consensus on held-out wording: not usable.
    assert verdict(table(0.85, (0.9, 0.9, 0.85), (0.5, 0.5, 0.5), 0.60)) == "human coders"
    # Same, but its mean kappa with each coder on held-out wording is .11 below that ceiling: not usable.
    assert verdict(table(0.85, (0.9, 0.9, 0.85), (0.5, 0.5, 0.5), 0.9, model_heldout_each=0.74)) == "human coders"
