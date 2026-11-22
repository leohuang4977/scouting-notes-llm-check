"""Generate synthetic amateur scouting notes with known true codes.

Why codes first and text second: each note starts as a set of true codes
(role, velocity, command, and so on), and the text is then written from
phrase banks to express exactly those codes. That is what makes the
question "when the model and the human coders disagree, who is right?"
answerable. With real notes nobody knows the true code.

The notes vary in some of the ways real ones do: jargon, hedges,
negations, missing information, a closing judgment that is sometimes
absent, and typos in about 5% of notes. About 15% are flagged ambiguous
and use at least one sentence that needs a codebook rule to resolve.
Test notes also use wording that never appears in dev notes, and each
note records which fields used it (see phrases.py).

Run:  python -m src.generate_notes        (writes data/notes.csv)
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src import phrases as P
from src.codebook import (
    DATA_DIR,
    N_NOTES,
    STREAM_NOTES,
    field_names,
    note_id,
    rng_for,
    split_of,
)

MIN_SENTENCES, MAX_SENTENCES = 2, 5
P_AMBIGUOUS = 0.15
P_TYPO = 0.05
P_PITCHER = 0.55

TONE_PROBS = {"negative": 0.25, "mixed": 0.50, "positive": 0.25}

# P(code | overall tone) for each field, including the "not mentioned" code.
# Tone comes first so that content and tone fit together, as in a real note.
FIELD_PROBS = {
    "velo_or_power": {
        "positive": {"plus": 0.50, "average": 0.25, "below_average": 0.05, "not_mentioned": 0.20},
        "mixed": {"plus": 0.25, "average": 0.30, "below_average": 0.25, "not_mentioned": 0.20},
        "negative": {"plus": 0.05, "average": 0.30, "below_average": 0.45, "not_mentioned": 0.20},
    },
    "command_or_approach": {
        "positive": {"strength": 0.35, "neutral": 0.20, "concern": 0.10, "not_mentioned": 0.35},
        "mixed": {"strength": 0.20, "neutral": 0.20, "concern": 0.25, "not_mentioned": 0.35},
        "negative": {"strength": 0.05, "neutral": 0.20, "concern": 0.40, "not_mentioned": 0.35},
    },
    "makeup": {
        "positive": {"positive": 0.30, "negative": 0.03, "not_mentioned": 0.67},
        "mixed": {"positive": 0.18, "negative": 0.12, "not_mentioned": 0.70},
        "negative": {"positive": 0.05, "negative": 0.25, "not_mentioned": 0.70},
    },
    "health_flag": {
        "positive": {"yes": 0.10, "no": 0.90},
        "mixed": {"yes": 0.20, "no": 0.80},
        "negative": {"yes": 0.30, "no": 0.70},
    },
    "projection": {
        "positive": {"high": 0.30, "some": 0.20, "limited": 0.05, "not_stated": 0.45},
        "mixed": {"high": 0.15, "some": 0.20, "limited": 0.15, "not_stated": 0.50},
        "negative": {"high": 0.05, "some": 0.15, "limited": 0.30, "not_stated": 0.50},
    },
}

# +1 favorable, -1 unfavorable. Used only to check that tone fits content.
VALENCE = {
    "velo_or_power": {"plus": 1, "below_average": -1},
    "command_or_approach": {"strength": 1, "concern": -1},
    "makeup": {"positive": 1, "negative": -1},
    "health_flag": {"yes": -1},
    "projection": {"high": 1, "limited": -1},
}

VELO_RANGES = {"below_average": [83, 84, 85], "average": [88, 89], "plus": [92, 93, 94]}


def pick(rng: np.random.Generator, options: list[str]) -> str:
    """One option, chosen uniformly. Returns a plain str, not a numpy str."""
    return options[int(rng.integers(len(options)))]


def draw_from(rng: np.random.Generator, probs: dict[str, float]) -> str:
    """One key of `probs`, drawn with the given probabilities."""
    keys = list(probs)
    weights = np.array([probs[k] for k in keys], dtype=float)
    return keys[int(rng.choice(len(keys), p=weights / weights.sum()))]


def draw_codes(rng: np.random.Generator) -> tuple[str, dict[str, str]]:
    """Draw a player type and the true code for every field."""
    player_type = "pitcher" if rng.random() < P_PITCHER else "position_player"
    if player_type == "pitcher":
        role = "starter" if rng.random() < 0.55 else "reliever"
    else:
        role = "position_player"
    tone = draw_from(rng, TONE_PROBS)
    codes = {"role": role, "overall_tone": tone}
    for field, by_tone in FIELD_PROBS.items():
        codes[field] = draw_from(rng, by_tone[tone])
    return player_type, codes


def tone_fits(codes: dict[str, str], has_closing: bool) -> bool:
    """True if the overall tone is defensible from the note's content.

    Mixed needs at least one favorable and one unfavorable point. Positive
    needs a favorable point and no unfavorable ones, unless a closing
    judgment settles it, in which case one unfavorable point is allowed.
    Negative is the mirror image. This keeps every true tone code
    consistent with the codebook rule for overall_tone.
    """
    marks = [VALENCE[f].get(codes[f], 0) for f in VALENCE]
    n_pos, n_neg = marks.count(1), marks.count(-1)
    slack = 1 if has_closing else 0
    if codes["overall_tone"] == "mixed":
        return n_pos >= 1 and n_neg >= 1
    if codes["overall_tone"] == "positive":
        return n_pos >= 1 and n_neg <= slack
    return n_neg >= 1 and n_pos <= slack


def hard_candidates(codes: dict[str, str], has_closing: bool) -> list[str]:
    """Fields that have a hard sentence available for their true code."""
    out = ["role", "makeup", "health_flag", "projection"]
    if codes["velo_or_power"] != "not_mentioned":
        out.append("velo_or_power")
    if codes["command_or_approach"] != "not_mentioned":
        out.append("command_or_approach")
    if has_closing:
        out.append("overall_tone")
    return out


def placeholders(player_type: str, codes: dict[str, str], rng: np.random.Generator) -> dict:
    """Values for the {braces} in phrase templates, consistent with the codes."""
    pos = pick(rng, list(P.POSITIONS))
    noun = P.POSITIONS[pos]
    feet_inches = pick(rng, ["5-11", "6-0", "6-1", "6-2", "6-3", "6-4", "6-5", "6-6"])
    fill = {
        "hand": "RHP" if rng.random() < 0.7 else "LHP",
        "ht": feet_inches,
        "wt": str(int(rng.integers(34, 48)) * 5),  # 170 to 235 in steps of 5
        "pos": pos,
        "noun": noun,
        "Noun": noun[0].upper() + noun[1:],
        "side": pick(rng, ["right", "left", "both"]),
        "asset": "arm" if player_type == "pitcher" else "bat",
        "lo": "", "hi": "", "peak": "",
    }
    velo = codes["velo_or_power"]
    if player_type == "pitcher" and velo in VELO_RANGES:
        lo = int(pick(rng, VELO_RANGES[velo]))
        hi = lo + 2
        # Peak only appears in plus sentences (a little above the range) and in
        # hard sentences (well above it, to tempt a coder to use the peak).
        bump = int(rng.integers(1, 3)) if velo == "plus" else int(rng.integers(3, 6))
        fill.update(lo=str(lo), hi=str(hi), peak=str(hi + bump))
    return fill


def render(
    player_type: str, codes: dict[str, str], plan: dict, split: str, rng: np.random.Generator
) -> tuple[list[str], set[str]]:
    """Turn true codes plus a plan (which optional pieces appear) into sentences.

    Returns the sentences and the set of fields whose sentence used held-out
    wording. Dev notes draw only from the base lists in phrases.py. Test
    notes draw from the base lists and their "_new" partners, so some test
    sentences use wording no dev note contains.
    """
    hard = plan["hard_fields"]
    heldout: set[str] = set()

    def choose(bank: dict, key: str, field: str) -> str:
        """Pick from bank[key], adding bank[key + '_new'] for test notes, and record held-out use."""
        base = bank[key]
        options = base + (bank[key + "_new"] if split == "test" else [])
        i = int(rng.integers(len(options)))
        if i >= len(base):
            heldout.add(field)
        return options[i]

    def kind(field: str) -> str:
        return "hard" if field in hard else "plain"

    opener = choose(P.OPENERS[codes["role"]], kind("role"), "role")
    if plan["projection_clause"]:
        clause = choose(P.PROJECTION[codes["projection"]], "clause", "projection")
        opener = opener[:-1] + ", " + clause + "."
    sentences = [opener]

    middle = []
    if plan["context"]:
        middle.append(pick(rng, P.CONTEXT))
    if codes["velo_or_power"] != "not_mentioned":
        bank = P.VELO if player_type == "pitcher" else P.POWER
        middle.append(choose(bank[codes["velo_or_power"]], kind("velo_or_power"), "velo_or_power"))
    if codes["command_or_approach"] != "not_mentioned":
        bank = P.COMMAND if player_type == "pitcher" else P.APPROACH
        middle.append(choose(bank[codes["command_or_approach"]], kind("command_or_approach"), "command_or_approach"))
    if codes["projection"] == "not_stated":
        if "projection" in hard:
            decoys = P.PROJECTION["not_stated"]
            by_role = {"hard": [decoys["hard_by_role"][codes["role"]]],
                       "hard_new": [decoys["hard_by_role_new"][codes["role"]]]}
            middle.append(choose(by_role, "hard", "projection"))
    elif not plan["projection_clause"]:
        middle.append(choose(P.PROJECTION[codes["projection"]], kind("projection"), "projection"))
    if codes["makeup"] != "not_mentioned" or "makeup" in hard:
        middle.append(choose(P.MAKEUP[codes["makeup"]], kind("makeup"), "makeup"))
    if codes["health_flag"] == "yes":
        style = "hard" if "health_flag" in hard else ("hedged" if plan["hedged_health"] else "plain")
        middle.append(choose(P.HEALTH["yes"], style, "health_flag"))
    elif "health_flag" in hard:
        style = "hard_pitcher" if player_type == "pitcher" else "hard_hitter"
        middle.append(choose(P.HEALTH["no"], style, "health_flag"))
    elif plan["health_negation"]:
        middle.append(choose(P.HEALTH["no"], "plain", "health_flag"))
    if rng.random() < 0.3:
        middle = [middle[i] for i in rng.permutation(len(middle))]
    sentences += middle

    if plan["closing"]:
        sentences.append(choose(P.CLOSING[codes["overall_tone"]], kind("overall_tone"), "overall_tone"))
    fill = placeholders(player_type, codes, rng)
    return [s.format(**fill) for s in sentences], heldout


def add_typo(text: str, rng: np.random.Generator) -> str:
    """Introduce one typo (swap, drop or double a letter) in a word of 5+ letters."""
    words = list(re.finditer(r"[A-Za-z]{5,}", text))
    m = words[int(rng.integers(len(words)))]
    w = m.group()
    i = int(rng.integers(1, len(w) - 1))
    how = int(rng.integers(3))
    if how == 0 and w[i] != w[i + 1]:
        new = w[:i] + w[i + 1] + w[i] + w[i + 2:]
    elif how == 2:
        new = w[:i] + w[i] + w[i:]
    else:
        new = w[:i] + w[i + 1:]
    return text[: m.start()] + new + text[m.end():]


def make_note(split: str, rng: np.random.Generator) -> dict:
    """Draw one note. Redraws until tone fits content and length is 2 to 5 sentences.

    Whether the note is ambiguous is decided before the loop, so redrawing
    for length does not change the share of ambiguous notes.
    """
    ambiguous = rng.random() < P_AMBIGUOUS
    while True:
        player_type, codes = draw_codes(rng)
        has_closing = rng.random() < 0.7
        if not tone_fits(codes, has_closing):
            continue
        hard_fields: list[str] = []
        if ambiguous:
            options = hard_candidates(codes, has_closing)
            k = 2 if rng.random() < 0.3 else 1
            hard_fields = [options[i] for i in sorted(rng.choice(len(options), size=k, replace=False))]
        plan = {
            "hard_fields": hard_fields,
            "closing": has_closing,
            "context": rng.random() < 0.2,
            "hedged_health": codes["health_flag"] == "yes" and rng.random() < 0.4,
            "health_negation": codes["health_flag"] == "no" and rng.random() < 0.25,
            "projection_clause": (
                codes["projection"] != "not_stated"
                and "projection" not in hard_fields
                and "role" not in hard_fields
                and rng.random() < 0.5
            ),
        }
        sentences, heldout = render(player_type, codes, plan, split, rng)
        if MIN_SENTENCES <= len(sentences) <= MAX_SENTENCES:
            break
    text = " ".join(sentences)
    has_typo = rng.random() < P_TYPO
    if has_typo:
        text = add_typo(text, rng)
    hedged = plan["hedged_health"] and "health_flag" not in hard_fields
    return {
        "player_type": player_type,
        "text": text,
        "n_sentences": len(sentences),
        "ambiguous": ambiguous,
        "hard_fields": ";".join(hard_fields),
        "hedged_health": hedged,
        "has_closing": has_closing,
        "heldout_fields": ";".join(f for f in field_names() if f in heldout),
        "has_typo": has_typo,
        **{f"true_{f}": codes[f] for f in field_names()},
    }


def generate_notes(n: int = N_NOTES) -> pd.DataFrame:
    """Generate n notes from the project seed. Same seed, same notes, every time."""
    rng = rng_for(STREAM_NOTES)
    rows = []
    for i in range(1, n + 1):
        nid = note_id(i)
        split = split_of(nid)
        rows.append({"note_id": nid, "split": split, **make_note(split, rng)})
    return pd.DataFrame(rows)


def main() -> None:
    notes = generate_notes()
    out = DATA_DIR / "notes.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    notes.to_csv(out, index=False, lineterminator="\n")
    print(f"Wrote {len(notes)} notes to {out.relative_to(DATA_DIR.parent)}")


if __name__ == "__main__":
    main()
