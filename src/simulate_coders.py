"""Simulate three human coders and their consensus code.

The simulated coders do not read the text. Each starts from the true code
and then makes the kinds of mistakes chosen for this simulation:

  Slips that depend on the sentence. Slips are rare on plain wording that
  also appears in dev notes, more common on wording held out of dev, and
  much more common on hard sentences. Projection and tone call for more
  judgment, so they slip more, and tone slips more when there is no
  closing judgment to go on.

  One habit each:
    coder_a  leans positive on tone (codes mixed as positive about 15% of the time)
    coder_b  misses hedged health mentions more often than the others
    coder_c  adds plain random noise (3% of codes replaced by a random code)

  One shared habit, planted on purpose and declared in PLAN.md: all three
  coders often miss hedged health mentions ("may have had some forearm
  tightness, not confirmed"), which the codebook says code "yes". The
  misses are correlated: some hedged mentions are easy to miss for
  everyone at once. So on this one kind of sentence the consensus is often
  wrong, and the disagreement audit has a known answer it should recover.

The consensus is the majority code. When all three disagree, a simulated
adjudicator picks the true code 80% of the time and coder_a's code
otherwise.

Run:  python -m src.simulate_coders      (needs data/notes.csv)
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from src.codebook import (
    DATA_DIR,
    HUMAN_CODERS,
    STREAM_CODERS,
    Field,
    load_codebook,
    read_notes,
    rng_for,
    sentence_type,
)

# Chance of a random slip, by the type of sentence that carries the field
# (see sentence_type in codebook.py), then scaled by field.
SLIP_BY_TYPE = {"plain": 0.03, "held_out": 0.07, "hard": 0.30, "hedged": 0.03}
FIELD_FACTOR = {
    "role": 0.8,
    "velo_or_power": 1.0,
    "command_or_approach": 1.2,
    "makeup": 1.0,
    "health_flag": 0.6,
    "projection": 1.6,
    "overall_tone": 1.4,
}
NO_CLOSING_TONE_FACTOR = 1.5       # tone is harder to read without a closing judgment
POSITION_PLAYER_ROLE_FACTOR = 0.2  # a hitter is rarely mistaken for a pitcher

# The shared habit on hedged health mentions.
P_EASY_TO_MISS = 0.4               # share of hedged mentions that are easy to miss for everyone
MISS_IF_EASY = {"coder_a": 0.75, "coder_b": 0.85, "coder_c": 0.75}
MISS_OTHERWISE = {"coder_a": 0.05, "coder_b": 0.25, "coder_c": 0.05}

CODER_A_TONE_LEAN = 0.15
CODER_C_NOISE = 0.03
ADJUDICATOR_PICKS_TRUTH = 0.80


def pick(rng: np.random.Generator, options: list[str]) -> str:
    return options[int(rng.integers(len(options)))]


def slip_rate(field: str, note: pd.Series) -> float:
    """Chance that a coder slips on this field of this note."""
    rate = SLIP_BY_TYPE[sentence_type(note, field)] * FIELD_FACTOR[field]
    if field == "overall_tone" and not note["has_closing"]:
        rate *= NO_CLOSING_TONE_FACTOR
    if field == "role" and note["true_role"] == "position_player":
        rate *= POSITION_PLAYER_ROLE_FACTOR
    return min(rate, 0.6)


def slip_value(field: Field, true: str, rng: np.random.Generator) -> str:
    """A plausible wrong code: the kind of mistake a person makes.

    On ordered fields most slips go to the next category on the scale, and
    some miss the mention entirely. On role, a pitcher's role gets swapped.
    """
    if field.name == "role":
        if true == "position_player":
            return pick(rng, ["starter", "reliever"])
        return "reliever" if true == "starter" else "starter"
    if field.type == "binary":
        return [v for v in field.values if v != true][0]
    if field.scale:
        off_scale = [v for v in field.values if v not in field.scale]
        if true in off_scale:
            return pick(rng, field.scale)
        if off_scale and rng.random() < 0.2:
            return off_scale[0]
        i = field.scale.index(true)
        neighbours = [field.scale[j] for j in (i - 1, i + 1) if 0 <= j < len(field.scale)]
        return pick(rng, neighbours)
    # Nominal field with a "not mentioned" value (makeup).
    mentioned = [v for v in field.values if v != "not_mentioned"]
    if true == "not_mentioned":
        return pick(rng, mentioned)
    if rng.random() < 0.5:
        return "not_mentioned"
    return pick(rng, [v for v in mentioned if v != true])


def code_note(
    coder: str, note: pd.Series, fields: list[Field], easy_to_miss: bool, rng: np.random.Generator
) -> dict[str, str]:
    """One coder's codes for one note: the truth, plus slips, plus the habits.

    `easy_to_miss` is drawn once per note and shared by all three coders,
    which is what makes their misses on hedged health mentions correlated.
    """
    codes = {}
    for field in fields:
        true = note[f"true_{field.name}"]
        if field.name == "health_flag" and note["hedged_health"]:
            miss = (MISS_IF_EASY if easy_to_miss else MISS_OTHERWISE)[coder]
            value = "no" if rng.random() < miss else true
        elif rng.random() < slip_rate(field.name, note):
            value = slip_value(field, true, rng)
        else:
            value = true
        if coder == "coder_a" and field.name == "overall_tone" and value == "mixed":
            if rng.random() < CODER_A_TONE_LEAN:
                value = "positive"
        if coder == "coder_c" and rng.random() < CODER_C_NOISE:
            value = pick(rng, field.values)
        codes[field.name] = value
    return codes


def consensus_code(values: list[str], true: str, coder_a_value: str, rng: np.random.Generator) -> tuple[str, bool]:
    """Majority code, or an adjudicated code when all three coders disagree.

    Returns (code, was_adjudicated).
    """
    top, count = Counter(values).most_common(1)[0]
    if count >= 2:
        return top, False
    return (true if rng.random() < ADJUDICATOR_PICKS_TRUTH else coder_a_value), True


def simulate(notes: pd.DataFrame) -> pd.DataFrame:
    """Code every note with the three coders and form the consensus.

    Returns a long sheet: one row per note per coder (coder_a, coder_b,
    coder_c, consensus), with one column per codebook field.
    """
    fields = load_codebook()
    rng = rng_for(STREAM_CODERS)
    rows = []
    for _, note in notes.iterrows():
        easy_to_miss = bool(note["hedged_health"]) and rng.random() < P_EASY_TO_MISS
        by_coder = {c: code_note(c, note, fields, easy_to_miss, rng) for c in HUMAN_CODERS}
        consensus, adjudicated = {}, []
        for field in fields:
            values = [by_coder[c][field.name] for c in HUMAN_CODERS]
            code, was_adjudicated = consensus_code(
                values, note[f"true_{field.name}"], by_coder["coder_a"][field.name], rng
            )
            consensus[field.name] = code
            if was_adjudicated:
                adjudicated.append(field.name)
        for coder in HUMAN_CODERS:
            rows.append({"note_id": note["note_id"], "coder": coder, **by_coder[coder], "adjudicated_fields": ""})
        rows.append({"note_id": note["note_id"], "coder": "consensus", **consensus,
                     "adjudicated_fields": ";".join(adjudicated)})
    return pd.DataFrame(rows)


def main() -> None:
    notes = read_notes()
    sheet = simulate(notes)
    out = DATA_DIR / "human_codes.csv"
    sheet.to_csv(out, index=False, lineterminator="\n")
    print(f"Wrote {len(sheet)} rows ({len(notes)} notes x 4 coders) to data/human_codes.csv")


if __name__ == "__main__":
    main()
