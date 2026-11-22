"""Score every coder on the test notes, with bootstrap intervals, and apply the decision rule.

Three tables come out:

  agreement_by_field.csv   For all test notes and again for the test notes whose
                           sentence for that field uses held-out wording:
                           - each coder against the consensus: percent agreement,
                             Cohen's kappa, Gwet's AC1, PABAK, and accuracy against
                             the true codes
                           - the keyword list and each model against each human
                             coder, averaged over the three coders
                           - the human ceiling: the three coders against each
                             other, averaged over the three pairs
                           - the consensus against the true codes
  disagreement_audit.csv   For the keyword list and each model: on the notes where
                           it differs from the consensus, how often it was right,
                           how often the consensus was right, and how often
                           neither. Broken down by field and by sentence type.
  decision_by_field.csv    The decision rule from PLAN.md, field by field.

Why "against each human coder" for the ceiling clause: a majority of three
coders is less noisy than any one coder, so anyone agrees better with the
consensus than with a single person. Comparing a machine's agreement with
the consensus to the humans' agreement with each other would flatter the
machine. So the ceiling clause compares like with like: the machine's mean
kappa with each human coder against the humans' mean kappa with each other.

Intervals: 1,000 bootstrap resamples of the test notes, seeded, percentile
95% intervals. The same resamples are used for every coder and field, so
comparisons between coders are paired.

Run:  python -m src.score --out outputs
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from src.codebook import (
    DATA_DIR,
    HUMAN_CODERS,
    REPO_ROOT,
    SENTENCE_TYPES,
    STREAM_BOOTSTRAP,
    Field,
    load_codebook,
    read_notes,
    read_sheet,
    rng_for,
    sentence_type,
)
from src.metrics import cohen_kappa, gwet_ac1, pabak, percent_agreement

N_BOOT = 1000
# Decision rule thresholds (see PLAN.md). Comparisons are inclusive.
BAR = 0.61             # kappa and AC1 with the consensus must both reach this
CEILING_GAP = 0.10     # mean kappa with each coder may be at most this far below the human ceiling
KEYWORD_MARGIN = 0.05  # the model must beat a passing keyword list by this much in kappa
SUBSETS = {"all": "all test notes", "held_out": "held-out wording"}
COUNT_KEYS = {"n", "n_disagree"}  # counts get no interval

Statistic = Callable[[np.ndarray], dict[str, float]]


# ---------------------------------------------------------------- helpers

def encode(sheet: pd.DataFrame, field: Field, note_ids: list[str]) -> np.ndarray:
    """Codes for one field as category numbers in codebook order; -1 where missing."""
    lookup = {value: i for i, value in enumerate(field.values)}
    values = sheet.set_index("note_id")[field.name].reindex(note_ids).fillna("")
    return np.array([lookup.get(v, -1) for v in values], dtype=int)


def bootstrap(statistic: Statistic, boot: np.ndarray, n: int) -> dict[str, float]:
    """Point estimate plus percentile 95% interval for every number the statistic returns.

    `statistic` takes an array of note positions and returns named numbers.
    It is run once on all notes (the point estimate) and once per resample.
    """
    point = statistic(np.arange(n))
    draws = [statistic(idx) for idx in boot]
    out: dict[str, float] = {}
    for key, value in point.items():
        out[key] = value
        if key in COUNT_KEYS:
            continue
        sample = np.array([d[key] for d in draws], dtype=float)
        if np.all(np.isnan(sample)):
            out[f"{key}_lo"] = out[f"{key}_hi"] = float("nan")
        else:
            out[f"{key}_lo"] = float(np.nanpercentile(sample, 2.5))
            out[f"{key}_hi"] = float(np.nanpercentile(sample, 97.5))
    return out


def agreement(x: np.ndarray, y: np.ndarray, q: int) -> dict[str, float]:
    """The four agreement numbers for two code arrays (nan if there are no notes)."""
    if len(x) == 0:
        return {"pct_agree": float("nan"), "kappa": float("nan"), "ac1": float("nan"), "pabak": float("nan")}
    return {"pct_agree": percent_agreement(x, y), "kappa": cohen_kappa(x, y, q),
            "ac1": gwet_ac1(x, y, q), "pabak": pabak(x, y, q)}


def mean_of(results: list[dict[str, float]]) -> dict[str, float]:
    """Average each named number over several results, ignoring undefined values."""
    return {k: float(np.nanmean([r[k] for r in results])) if not all(np.isnan(r[k]) for r in results)
            else float("nan") for k in results[0]}


# ---------------------------------------------------------------- statistics

def pair_statistic(a: np.ndarray, b: np.ndarray, truth: np.ndarray, q: int, keep: np.ndarray) -> Statistic:
    """Agreement of a with b, and accuracy of a against the truth, on the kept notes."""

    def statistic(idx: np.ndarray) -> dict[str, float]:
        x, y, t, k = a[idx], b[idx], truth[idx], keep[idx]
        ok = k & (x >= 0) & (y >= 0)
        out = {"n": float(ok.sum()), **agreement(x[ok], y[ok], q)}
        out["accuracy_vs_truth"] = percent_agreement(x[ok], t[ok]) if ok.any() else float("nan")
        return out

    return statistic


def each_coder_statistic(machine: np.ndarray, coders: list[np.ndarray], q: int, keep: np.ndarray) -> Statistic:
    """A machine coder's agreement with each human coder, averaged over the three."""

    def statistic(idx: np.ndarray) -> dict[str, float]:
        m, k = machine[idx], keep[idx]
        ok = k & (m >= 0)
        return {"n": float(ok.sum()), **mean_of([agreement(m[ok], c[idx][ok], q) for c in coders])}

    return statistic


def ceiling_statistic(coders: list[np.ndarray], q: int, keep: np.ndarray) -> Statistic:
    """Mean pairwise agreement among the human coders: the human ceiling."""

    def statistic(idx: np.ndarray) -> dict[str, float]:
        k = keep[idx]
        pairs = [agreement(x[idx][k], y[idx][k], q) for x, y in combinations(coders, 2)]
        return {"n": float(k.sum()), **mean_of(pairs)}

    return statistic


def audit_statistic(
    coder: list[np.ndarray], consensus: list[np.ndarray], truth: list[np.ndarray], keep: list[np.ndarray]
) -> Statistic:
    """Who is right when a coder and the consensus disagree. Lists hold one array per field."""

    def statistic(idx: np.ndarray) -> dict[str, float]:
        c = np.concatenate([x[idx][m[idx]] for x, m in zip(coder, keep)])
        k = np.concatenate([x[idx][m[idx]] for x, m in zip(consensus, keep)])
        t = np.concatenate([x[idx][m[idx]] for x, m in zip(truth, keep)])
        differ = (c >= 0) & (c != k)
        if not differ.any():
            return {"n_disagree": 0.0, "share_coder_right": float("nan"),
                    "share_consensus_right": float("nan"), "share_neither": float("nan")}
        c, k, t = c[differ], k[differ], t[differ]
        return {"n_disagree": float(differ.sum()),
                "share_coder_right": float(np.mean(c == t)),
                "share_consensus_right": float(np.mean(k == t)),
                "share_neither": float(np.mean((c != t) & (k != t)))}

    return statistic


# ---------------------------------------------------------------- tables

def score(
    notes: pd.DataFrame,
    human: pd.DataFrame,
    machine_sheets: dict[str, pd.DataFrame],
    n_boot: int = N_BOOT,
) -> dict[str, pd.DataFrame]:
    """Build the three tables on the test split.

    machine_sheets maps a label ("keyword", a model name, or "MOCK") to its
    code sheet. The keyword sheet must be labelled "keyword".
    """
    fields = load_codebook()
    test = notes[notes["split"] == "test"].sort_values("note_id")
    ids = test["note_id"].tolist()
    n = len(ids)
    boot = rng_for(STREAM_BOOTSTRAP).integers(0, n, size=(n_boot, n))
    truth_sheet = test.rename(columns={f"true_{f.name}": f.name for f in fields})

    sheets = {c: human[human["coder"] == c] for c in [*HUMAN_CODERS, "consensus"]}
    sheets.update(machine_sheets)
    codes = {label: {f.name: encode(s, f, ids) for f in fields} for label, s in sheets.items()}
    truth = {f.name: encode(truth_sheet, f, ids) for f in fields}
    types = {f.name: np.array([sentence_type(row, f.name) for _, row in test.iterrows()]) for f in fields}
    keep = {
        "all": {f.name: np.ones(n, dtype=bool) for f in fields},
        "held_out": {f.name: test["heldout_fields"].str.split(";").map(lambda s, f=f: f.name in s).to_numpy()
                     for f in fields},
    }

    def kind(label: str) -> str:
        if label in HUMAN_CODERS:
            return "human"
        return "keyword" if label == "keyword" else "model"

    rows = []
    for subset, label_text in SUBSETS.items():
        for f in fields:
            q, kp = len(f.values), keep[subset][f.name]
            cons, humans = codes["consensus"][f.name], [codes[c][f.name] for c in HUMAN_CODERS]
            base = {"field": f.name, "notes": label_text}
            people = HUMAN_CODERS if subset == "all" else []
            for label in [*people, *machine_sheets]:
                stat = pair_statistic(codes[label][f.name], cons, truth[f.name], q, kp)
                rows.append({**base, "coder": label, "kind": kind(label), "reference": "consensus",
                             **bootstrap(stat, boot, n)})
            for label in machine_sheets:
                stat = each_coder_statistic(codes[label][f.name], humans, q, kp)
                rows.append({**base, "coder": label, "kind": kind(label), "reference": "each human coder (mean)",
                             **bootstrap(stat, boot, n)})
            stat = pair_statistic(cons, truth[f.name], truth[f.name], q, kp)
            rows.append({**base, "coder": "consensus", "kind": "consensus", "reference": "truth",
                         **bootstrap(stat, boot, n)})
            rows.append({**base, "coder": "human_ceiling", "kind": "human_ceiling",
                         "reference": "human coder pairs (mean)",
                         **bootstrap(ceiling_statistic(humans, q, kp), boot, n)})

    audit = []
    for label in machine_sheets:
        for scope in [*[f.name for f in fields], "all_fields"]:
            chosen = [f.name for f in fields] if scope == "all_fields" else [scope]
            for stype in ["all", *SENTENCE_TYPES]:
                masks = [np.ones(n, dtype=bool) if stype == "all" else types[x] == stype for x in chosen]
                if not any(m.any() for m in masks):
                    continue
                stat = audit_statistic([codes[label][x] for x in chosen], [codes["consensus"][x] for x in chosen],
                                       [truth[x] for x in chosen], masks)
                audit.append({"coder": label, "kind": kind(label), "field": scope, "sentence_type": stype,
                              **bootstrap(stat, boot, n)})

    agreement_df = pd.DataFrame(rows)
    return {
        "agreement_by_field": agreement_df,
        "disagreement_audit": pd.DataFrame(audit),
        "decision_by_field": decide(agreement_df, [m for m in machine_sheets if m != "keyword"]),
    }


def lookup(agreement_df: pd.DataFrame, field: str, coder: str, reference: str, notes: str, column: str) -> float:
    """One number from the agreement table."""
    row = agreement_df[(agreement_df["field"] == field) & (agreement_df["coder"] == coder)
                       & (agreement_df["reference"] == reference) & (agreement_df["notes"] == notes)]
    return float(row[column].iloc[0])


def at_least(a: float, b: float) -> bool:
    """a >= b, allowing for floating-point rounding (so .80 + .05 counts as .85). nan is never at least anything."""
    return bool(a >= b - 1e-9)


def meets_bar(kappa_consensus: float, ac1_consensus: float, kappa_each_coder: float, ceiling: float) -> bool:
    """The PLAN.md bar. Kappa and AC1 with the consensus both >= .61, and mean kappa
    with each human coder no more than .10 below the human ceiling. An undefined
    (nan) statistic does not meet the bar."""
    return (at_least(kappa_consensus, BAR) and at_least(ac1_consensus, BAR)
            and at_least(kappa_each_coder, ceiling - CEILING_GAP))


def decide(agreement_df: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Apply the decision rule to every field for every model."""
    rows = []
    for field in dict.fromkeys(agreement_df["field"]):

        def get(coder: str, reference: str, notes: str, column: str = "kappa") -> float:
            return lookup(agreement_df, field, coder, reference, notes, column)

        all_, ho = SUBSETS["all"], SUBSETS["held_out"]
        ceiling = get("human_ceiling", "human coder pairs (mean)", all_)
        ceiling_ho = get("human_ceiling", "human coder pairs (mean)", ho)
        kw = {"k": get("keyword", "consensus", ho), "a": get("keyword", "consensus", ho, "ac1"),
              "each": get("keyword", "each human coder (mean)", ho)}
        keyword_ok = meets_bar(kw["k"], kw["a"], kw["each"], ceiling_ho)
        for model in models:
            m_k, m_a = get(model, "consensus", all_), get(model, "consensus", all_, "ac1")
            m_each = get(model, "each human coder (mean)", all_)
            m_k_ho = get(model, "consensus", ho)
            m_a_ho = get(model, "consensus", ho, "ac1")
            m_each_ho = get(model, "each human coder (mean)", ho)
            # The model has to meet the bar on all test notes AND on held-out wording,
            # so a model that only echoes familiar phrasing is not called usable.
            model_ok = (meets_bar(m_k, m_a, m_each, ceiling)
                        and meets_bar(m_k_ho, m_a_ho, m_each_ho, ceiling_ho))
            keyword_preferred = keyword_ok and not at_least(m_k_ho, kw["k"] + KEYWORD_MARGIN)
            if not at_least(ceiling, BAR):
                verdict = "codebook needs work"
            elif keyword_preferred:
                verdict = "keyword list"
            elif model_ok:
                verdict = "MOCK coder" if model == "MOCK" else "model"
            elif keyword_ok:
                verdict = "keyword list"
            else:
                verdict = "human coders"
            rows.append({
                "field": field, "model": model,
                "human_ceiling_kappa": ceiling, "codebook_ok": at_least(ceiling, BAR),
                "model_kappa_consensus": m_k, "model_ac1_consensus": m_a,
                "model_kappa_each_coder": m_each, "model_gap_below_ceiling": ceiling - m_each,
                "model_usable": model_ok,
                "heldout_human_ceiling_kappa": ceiling_ho,
                "keyword_heldout_kappa_consensus": kw["k"], "keyword_heldout_ac1_consensus": kw["a"],
                "keyword_heldout_kappa_each_coder": kw["each"], "keyword_meets_bar_heldout": keyword_ok,
                "model_heldout_kappa_consensus": m_k_ho, "model_heldout_ac1_consensus": m_a_ho,
                "model_heldout_kappa_each_coder": m_each_ho,
                "keyword_kappa_consensus_all": get("keyword", "consensus", all_),
                "keyword_preferred": keyword_preferred, "verdict": verdict,
            })
    return pd.DataFrame(rows)


def write_tables(tables: dict[str, pd.DataFrame], out_dir: Path) -> None:
    """Write each table as CSV, numbers rounded to 3 decimals."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table = table.round(3)
        for col in COUNT_KEYS & set(table.columns):
            table[col] = table[col].astype(int)
        table.to_csv(out_dir / f"{name}.csv", index=False, lineterminator="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score the coders on the test notes.")
    parser.add_argument("--out", default="outputs", help="folder with keyword and model code sheets")
    args = parser.parse_args()
    out_dir = REPO_ROOT / args.out
    machine = {"keyword": read_sheet(out_dir / "keyword_codes.csv")}
    for path in sorted(out_dir.glob("model_codes_*.csv")):
        sheet = read_sheet(path)
        machine[sheet["coder"].iloc[0]] = sheet
    tables = score(read_notes(), read_sheet(DATA_DIR / "human_codes.csv"), machine)
    write_tables(tables, out_dir)
    print(f"Wrote agreement, audit and decision tables to {args.out}/")


if __name__ == "__main__":
    main()
