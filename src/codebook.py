"""Shared settings: the codebook, the seed, the dev/test split and repo paths.

Every other module imports from here, so the split rule and the seed are
written down in one place. If they lived in several files they could drift
apart, and a note could end up in dev for one step and test for another.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
CODEBOOK_PATH = Path(__file__).resolve().parent / "codebook.yaml"

# One project seed. Each step draws from its own stream of this seed
# (see rng_for), so changing how one step uses randomness does not shift
# the random draws of any other step.
SEED = 20261122
STREAM_NOTES = 1
STREAM_CODERS = 2
STREAM_MOCK = 3
STREAM_BOOTSTRAP = 4

N_NOTES = 600
N_DEV = 150  # notes N0001 to N0150 are dev; the rest are test

HUMAN_CODERS = ["coder_a", "coder_b", "coder_c"]


@dataclass(frozen=True)
class Field:
    """One codebook field. `scale` is empty for nominal and binary fields."""

    name: str
    type: str
    values: list[str]
    scale: list[str]
    definition: str
    rules: list[str]


def load_codebook(path: Path = CODEBOOK_PATH) -> list[Field]:
    """Read codebook.yaml into a list of Field objects, in file order."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    fields = []
    for name, spec in raw["fields"].items():
        fields.append(
            Field(
                name=name,
                type=spec["type"],
                values=[str(v) for v in spec["values"]],
                scale=[str(v) for v in spec.get("scale", [])],
                definition=" ".join(spec["definition"].split()),
                rules=[" ".join(r.split()) for r in spec["rules"]],
            )
        )
    return fields


def field_names() -> list[str]:
    """Field names in codebook order. Code sheets use this column order."""
    return [f.name for f in load_codebook()]


def note_id(number: int) -> str:
    """Note ids are N0001 to N0600, so sorting by id is sorting by number."""
    return f"N{number:04d}"


def split_of(note: str) -> str:
    """Return 'dev' for the first N_DEV notes by id and 'test' for the rest.

    The split depends only on the id, so it cannot be influenced by how a
    note reads or by any coder's output.
    """
    number = int(note.lstrip("N"))
    return "dev" if number <= N_DEV else "test"


def sheet_columns() -> list[str]:
    """Columns of every code sheet (human, keyword, model): one row per note per coder."""
    return ["note_id", "coder", *field_names()]


def read_notes(path: Path = DATA_DIR / "notes.csv") -> pd.DataFrame:
    """Read the notes file, turning the True/False columns back into booleans.

    Reading them as text would be a trap: the string "False" is truthy.
    """
    notes = pd.read_csv(path, dtype=str, keep_default_na=False)
    for col in ["ambiguous", "hedged_health", "has_typo", "has_closing"]:
        notes[col] = notes[col].eq("True")
    notes["n_sentences"] = notes["n_sentences"].astype(int)
    return notes


def read_sheet(path: Path) -> pd.DataFrame:
    """Read a code sheet with every value kept as text (so "no" stays "no")."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


# Sentence types, used both to set how often the simulated coders slip and
# to break the disagreement audit down.
SENTENCE_TYPES = {
    "plain": "Plain wording seen in dev",
    "held_out": "Wording held out of dev",
    "hard": "Hard sentence (ambiguous note)",
    "hedged": "Hedged health mention",
}


def sentence_type(note: pd.Series, field: str) -> str:
    """The type of the sentence that carries `field` in this note.

    A hedged health mention counts as hedged even if its wording is also
    held out. A field written with a hard sentence counts as hard. A field
    the note does not mention counts as plain.
    """
    if field == "health_flag" and note["hedged_health"]:
        return "hedged"
    if field in str(note["hard_fields"]).split(";"):
        return "hard"
    if field in str(note["heldout_fields"]).split(";"):
        return "held_out"
    return "plain"


def rng_for(stream: int, *extra: int) -> np.random.Generator:
    """A numpy Generator for one step of the pipeline, seeded from SEED.

    numpy accepts a list of integers as a seed, so [SEED, stream] gives each
    step its own independent stream while the whole project still depends on
    one number.
    """
    return np.random.default_rng([SEED, stream, *extra])
