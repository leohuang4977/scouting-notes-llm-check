"""Keyword baseline: code each note with hand-written regular expressions.

Why have a baseline at all: a language model is only worth its cost on a
field if it beats something cheap and transparent. If a short regex list
codes velocity as well as the model does, the club should use the regex.

How the list was built: from the 150 dev notes only. Dev notes never
contain the held-out phrasings that test notes use (see phrases.py), so
the list meets new wording on test, as it would on a new season of
notes. tests/test_keywords.py checks that every phrase below appears in
at least one dev note.

How it codes a field: split the note into sentences, keep the sentences
that are about the field (its "topic"), and try the field's rules in
order on those sentences. The first rule that matches gives the code. If
nothing matches, the field gets its default code.

One caveat stays: the same author (Claude) wrote the phrase banks and
this list, so on familiar wording its scores are a generous estimate of
what a keyword list would do on real notes. The comparison on held-out
wording is the fairer one.

Run:  python -m src.code_keywords --out outputs
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from src.codebook import REPO_ROOT, read_notes, sheet_columns

CODER_NAME = "keyword"

# Each field: a topic pattern that picks its sentences (None = whole note),
# ordered (code, pattern) rules, and a default code.
# Order matters: "below-average" contains "average", so below comes first.
FIELD_RULES: dict[str, dict] = {
    "role": {
        "topic": None,
        "rules": [
            ("position_player", r"^(C|1B|2B|3B|SS|LF|CF|RF)\b"),
            ("position_player", r"(?i)catcher|shortstop|baseman|fielder|two-way"),
            ("reliever", r"(?i)relie(f|ver)|bullpen|out of the pen"),
            ("starter", r"(?i)starter|started|rotation|six innings"),
        ],
        "default": "starter",
    },
    "velo_or_power": {
        "topic": r"(?i)fastball|velocity|heater|power|\braw\b|\bpop\b|\bsits\b|\bsat\b",
        "rules": [
            # A velocity range like "89 to 91" is handled first, in velo_from_range.
            ("below_average", r"(?i)below-average|below average|fringy|plays light|power is light|\ba (40|45)\b"),
            ("plus", r"(?i)plus|above-average|mid 90s|\ba (55|60|65|70)\b"),
            ("average", r"(?i)average|\ba 50\b"),
        ],
        "default": "not_mentioned",
    },
    "command_or_approach": {
        "topic": r"(?i)command|control|strikes|zone|walk|chase|approach|swing|pitch recognition|counts|behind|paints|locating",
        "skip": r"(?i)fastball|heater|velocity",  # "life up in the zone" is about the fastball
        "rules": [
            ("strength", r"(?i)never seems to fall behind|rarely expands"),
            ("concern", r"(?i)walks too many|spotty|below average|below-average|falls behind|chases|expands|free swinger|approach is raw|swing and miss|not quality"),
            ("strength", r"(?i)plus|advanced|pounds the zone|rarely|paints|works counts|controls the zone"),
            ("neutral", r"(?i)enough|average|fine|nothing alarming"),
        ],
        "default": "not_mentioned",
    },
    "makeup": {
        "topic": r"(?i)makeup|work ethic|character|competes|leader|teammates|coach|kid\b|first one at the field|body language|adjustments",
        "rules": [
            ("negative", r"(?i)questions|concerns|hard to coach|body language sags|does not put in the work"),
            ("positive", r"(?i)rave|high-character|plus makeup|leader|competes|first one"),
        ],
        "default": "not_mentioned",
    },
    "health_flag": {
        "topic": None,
        "rules": [
            ("no", r"(?i)no injury history|medicals are clean|nothing that worries us physically|only a precaution"),
            ("yes", r"(?i)missed|surgery|sore|tightness|shut down|strain|cramping"),
        ],
        "default": "no",
    },
    "projection": {
        "topic": r"(?i)projection|projectable|room|frame|body|filled out|maxed|mature|strength",
        "rules": [
            ("limited", r"(?i)maxed out|filled out|physically mature|not much projection|what you see|already 23"),
            ("high", r"(?i)very projectable|plenty of projection|lots of room|plenty of strength"),
            ("some", r"(?i)some projection|bit more strength|moderate room"),
        ],
        "default": "not_stated",
    },
    "overall_tone": {
        "topic": None,
        "rules": [
            ("mixed", r"(?i)some tools, some holes|wait-and-see|tough read|good and bad|intriguing but raw|another look"),
            ("negative", r"(?i)\bpass\b|no follow|not a follow|hard to see a path|would not spend"),
            ("positive", r"(?i)follow closely|strong follow|priority follow|worth pushing|hard to walk away"),
        ],
        "default": None,  # no closing phrase found: fall back to counting good and bad words
    },
}

# Used only when no closing phrase decides overall_tone.
FAVORABLE = r"(?i)plus|above-average|advanced|rave|high-character|projectable|pounds the zone|patient"
UNFAVORABLE = r"(?i)below|fringy|walks too many|spotty|surgery|missed|sore|questions|concerns|chases|expands"

VELO_RANGE = re.compile(r"\b(\d{2}) to (\d{2})\b")


def sentences_of(text: str) -> list[str]:
    """Split a note into sentences. The generator never puts a period inside a sentence."""
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


def velo_from_range(sentence: str) -> str | None:
    """Code a pitcher's velocity from 'sits 89 to 91', using the low end of the range.

    The codebook says to code where the fastball sits. Using the range and
    ignoring single peak numbers ("touching 96") follows that rule.
    """
    m = VELO_RANGE.search(sentence)
    if not m:
        return None
    low = int(m.group(1))
    if low < 88:
        return "below_average"
    return "average" if low <= 91 else "plus"


def code_field(field: str, text: str) -> str:
    """Code one field of one note with the rules above."""
    spec = FIELD_RULES[field]
    if spec["topic"] is None:
        candidates = [text]
    else:
        skip = spec.get("skip")
        candidates = [
            s for s in sentences_of(text)
            if re.search(spec["topic"], s) and not (skip and re.search(skip, s))
        ]
    for sentence in candidates:
        if field == "velo_or_power":
            from_range = velo_from_range(sentence)
            if from_range:
                return from_range
        for code, pattern in spec["rules"]:
            if re.search(pattern, sentence):
                return code
    if field == "overall_tone":
        return tone_from_word_counts(text)
    return spec["default"]


def tone_from_word_counts(text: str) -> str:
    """Fallback for tone: both kinds of words is mixed, one kind decides, none is mixed."""
    good = len(re.findall(FAVORABLE, text))
    bad = len(re.findall(UNFAVORABLE, text))
    if good and not bad:
        return "positive"
    if bad and not good:
        return "negative"
    return "mixed"


def code_notes(notes: pd.DataFrame) -> pd.DataFrame:
    """Code every note. Returns a code sheet with coder = 'keyword'."""
    rows = []
    for _, note in notes.iterrows():
        row = {"note_id": note["note_id"], "coder": CODER_NAME}
        for field in FIELD_RULES:
            row[field] = code_field(field, note["text"])
        rows.append(row)
    return pd.DataFrame(rows, columns=sheet_columns())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="outputs", help="output folder (relative to the repo)")
    args = parser.parse_args()
    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    sheet = code_notes(read_notes())
    path: Path = out_dir / "keyword_codes.csv"
    sheet.to_csv(path, index=False, lineterminator="\n")
    print(f"Wrote keyword codes for {len(sheet)} notes to {args.out}/keyword_codes.csv")


if __name__ == "__main__":
    main()
