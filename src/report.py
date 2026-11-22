"""Write outputs/results.md: was the plan committed first, the run details,
a five-line summary, and tables by field and by sentence type.

Kept apart from run_all.py so the run script stays a short list of steps.
"""

from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.codebook import REPO_ROOT, SEED, SENTENCE_TYPES, field_names
from src.figure import LABELS
from src.score import N_BOOT, SUBSETS

PLAN_FILES = {
    "PLAN.md": "the plan and decision rule",
    "src/codebook.yaml": "the codebook the prompt is built from",
    "src/code_llm.py": "the code that builds the prompt",
}


# ---------------------------------------------------------------- git

def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True)


def prompt_hash_in(plan_text: str) -> str | None:
    m = re.search(r"Prompt hash: `([0-9a-f]{12})`", plan_text)
    return m.group(1) if m else None


def git_state() -> dict:
    """What git can show about the plan at the start of a run.

    Returns the last commit and, for each planning file, whether it changed
    since that commit. If there is no repository (or no git), says so, so
    the results page never implies a check that did not happen.
    """
    try:
        top = git("rev-parse", "--show-toplevel")
    except FileNotFoundError:
        return {"commit": None, "reason": "git is not installed"}
    if top.returncode != 0:
        return {"commit": None, "reason": "there is no git repository here"}
    if Path(top.stdout.strip()).resolve() != REPO_ROOT.resolve():
        return {"commit": None, "reason": "this folder sits inside another git repository instead of being its own"}
    head = git("rev-parse", "HEAD")
    if head.returncode != 0:
        return {"commit": None, "reason": "the git repository has no commit yet"}
    files = {}
    for path in PLAN_FILES:
        if git("ls-files", "--error-unmatch", path).returncode != 0:
            files[path] = "not in the commit"
        else:
            files[path] = "changed since the commit" if git("diff", "--quiet", "HEAD", "--", path).returncode else "unchanged"
    committed_plan = git("show", "HEAD:PLAN.md")
    return {
        "commit": head.stdout.strip(),
        "commit_time": git("show", "-s", "--format=%ci", "HEAD").stdout.strip(),
        "files": files,
        "plan_prompt_hash": prompt_hash_in(committed_plan.stdout) if committed_plan.returncode == 0 else None,
    }


def git_lines(state: dict, prompt_hashes: list[str]) -> list[str]:
    if state["commit"] is None:
        return [f"No git commit to check against ({state['reason']}), so this run cannot show that the plan "
                "was written down before it ran."]
    lines = [f"- Last commit before this run: {state['commit']} ({state['commit_time']})."]
    for path, status in state["files"].items():
        lines.append(f"- `{path}` ({PLAN_FILES[path]}): {status}.")
    for h in dict.fromkeys(prompt_hashes):
        same = "matches" if h == state["plan_prompt_hash"] else "DOES NOT match"
        lines.append(f"- Prompt hash used in this run, {h}, {same} the hash in PLAN.md at that commit "
                     f"({state['plan_prompt_hash']}).")
    return lines


# ---------------------------------------------------------------- numbers

def fmt(x: float) -> str:
    """Two decimals without the leading zero, as is usual for kappa: 0.83 -> .83."""
    if pd.isna(x):
        return "n/a"
    text = f"{x:.2f}"
    return text.replace("0.", ".", 1) if abs(x) < 1 else text


def pct(x: float) -> str:
    return "n/a" if pd.isna(x) else f"{x:.0%}"


def pick(table: pd.DataFrame, coder: str, reference: str, notes: str = SUBSETS["all"]) -> pd.DataFrame:
    rows = table[(table["coder"] == coder) & (table["reference"] == reference) & (table["notes"] == notes)]
    return rows.set_index("field")


def summary_lines(model: str, tables: dict[str, pd.DataFrame]) -> list[str]:
    """The five-line summary for one model, filled from the tables."""
    agree, audit, decisions = tables["agreement_by_field"], tables["disagreement_audit"], tables["decision_by_field"]
    d = decisions[decisions["model"] == model]
    who = "the MOCK coder" if model == "MOCK" else "the model"  # never call MOCK output a model

    def named(mask: pd.Series) -> str:
        names = [LABELS[f].lower() for f in d.loc[mask, "field"]]
        return ", ".join(names) if names else "none"

    each = "each human coder (mean)"
    ceiling = pick(agree, "human_ceiling", "human coder pairs (mean)")["kappa"].median()
    ho = SUBSETS["held_out"]
    audit_m = audit[(audit["coder"] == model) & (audit["field"] == "all_fields")].set_index("sentence_type")
    pooled, hedged = audit_m.loc["all"], audit_m.loc["hedged"]
    verdict = d["verdict"]
    return [
        f"1. Under the rule in PLAN.md, {who} meets the bar on {int(d['model_usable'].sum())} of {len(d)} fields: "
        f"{named(d['model_usable'])}.",
        f"2. Verdict by field. {who[0].upper() + who[1:]}: {named(verdict.isin(['model', 'MOCK coder']))}. "
        f"Keyword list: {named(verdict == 'keyword list')}. Human coders still needed: "
        f"{named(verdict == 'human coders')}. Codebook needs work: {named(verdict == 'codebook needs work')}.",
        f"3. Mean kappa with each human coder, median across fields: {who} "
        f"{fmt(pick(agree, model, each)['kappa'].median())}, keyword list {fmt(pick(agree, 'keyword', each)['kappa'].median())}. "
        f"The human coders with each other: {fmt(ceiling)}.",
        f"4. On wording held out of the dev notes, median kappa with the consensus across fields: {who} "
        f"{fmt(pick(agree, model, 'consensus', ho)['kappa'].median())}, keyword list "
        f"{fmt(pick(agree, 'keyword', 'consensus', ho)['kappa'].median())} (on all test notes: "
        f"{fmt(pick(agree, model, 'consensus')['kappa'].median())} and "
        f"{fmt(pick(agree, 'keyword', 'consensus')['kappa'].median())}).",
        f"5. Where {who} and the consensus disagreed ({int(pooled['n_disagree'])} codes), {who} matched the true "
        f"code {pct(pooled['share_coder_right'])} of the time and the consensus {pct(pooled['share_consensus_right'])}. "
        f"On hedged health mentions, where the simulated coders share a planted habit, {who} was right in "
        f"{pct(hedged['share_coder_right'])} of {int(hedged['n_disagree'])} disagreements.",
    ]


def field_table(model: str, tables: dict[str, pd.DataFrame]) -> list[str]:
    """Main numbers by field for one model."""
    agree, decisions = tables["agreement_by_field"], tables["decision_by_field"]
    d = decisions[decisions["model"] == model].set_index("field")
    cons, each = pick(agree, model, "consensus"), pick(agree, model, "each human coder (mean)")
    ceiling = pick(agree, "human_ceiling", "human coder pairs (mean)")
    kw_all, kw_ho = pick(agree, "keyword", "consensus"), pick(agree, "keyword", "consensus", SUBSETS["held_out"])
    m_ho = pick(agree, model, "consensus", SUBSETS["held_out"])
    who = "MOCK" if model == "MOCK" else "Model"
    lines = [
        f"| Field | {who} kappa with consensus [95% CI] | {who} AC1 with consensus | {who} mean kappa with each coder "
        f"| Human coders with each other | Keyword kappa, all / held-out | {who} kappa, held-out | Verdict |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for f in field_names():
        lines.append(
            f"| {LABELS[f]} | {fmt(cons.loc[f, 'kappa'])} [{fmt(cons.loc[f, 'kappa_lo'])}, {fmt(cons.loc[f, 'kappa_hi'])}] "
            f"| {fmt(cons.loc[f, 'ac1'])} | {fmt(each.loc[f, 'kappa'])} | {fmt(ceiling.loc[f, 'kappa'])} "
            f"| {fmt(kw_all.loc[f, 'kappa'])} / {fmt(kw_ho.loc[f, 'kappa'])} | {fmt(m_ho.loc[f, 'kappa'])} "
            f"| {d.loc[f, 'verdict']} |"
        )
    return lines


def audit_table(model: str, tables: dict[str, pd.DataFrame]) -> list[str]:
    """Who was right when the model and the consensus disagreed, by sentence type."""
    audit = tables["disagreement_audit"]
    rows = audit[(audit["coder"] == model) & (audit["field"] == "all_fields")].set_index("sentence_type")
    who = "MOCK" if model == "MOCK" else "Model"
    lines = [f"| Sentence type | Disagreements | {who} right | Consensus right | Neither |", "|---|---|---|---|---|"]
    for stype, label in [*SENTENCE_TYPES.items(), ("all", "All sentences")]:
        r = rows.loc[stype]
        lines.append(f"| {label} | {int(r['n_disagree'])} | {pct(r['share_coder_right'])} "
                     f"| {pct(r['share_consensus_right'])} | {pct(r['share_neither'])} |")
    return lines


FIGURE_TEXT = [
    "Figure (`figures/agreement_by_field.png`). Left: for each field, the keyword list's and the model's mean",
    "Cohen's kappa with each of the three human coders, beside the human coders' mean kappa with each other,",
    "with bootstrap 95% intervals. The grey tick marks the human value minus .10, the threshold in the",
    "decision rule. Right: on the test notes where the model and the coders' consensus disagree, the share",
    "where the model matched the true code, where the consensus did, and where neither did, by the type of",
    "sentence the field was written in.",
]


def write_results(out_dir: Path, tables: dict[str, pd.DataFrame], manifests: list[dict],
                  git_info: dict, mock: bool) -> Path:
    """Write results.md and return its path."""
    lines = ["# Results", ""]
    if mock:
        lines += ["**MOCK RUN. The 'model' below is the true codes plus random noise, used only to test",
                  "the pipeline without Ollama. No number on this page is a language model result.**", ""]
    when = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines += [f"Run finished {when}. Test notes: 450. Bootstrap: {N_BOOT:,} resamples (seed {SEED}).", "",
              "## Was the plan written down before this run?", "",
              *git_lines(git_info, [m["prompt_hash"] for m in manifests]), "", "## Coders", ""]
    for m in manifests:
        lines += [
            f"- {m['model']} ({m['label']}). Digest: {m['model_digest']}. Ollama: {m['ollama_version']}.",
            f"  Options: {json.dumps(m['options'])}. Thinking: {m['thinking']}.",
            f"  Notes coded: {m['n_coded']} of {m['n_notes']}. Failed: {m['n_failed']}. "
            f"Median seconds per call: {m['median_seconds_per_call']}.",
        ]
    for m in manifests:
        model = m["model"]
        lines += ["", f"## Summary for {model}", "", *summary_lines(model, tables),
                  "", f"## By field for {model}", "", *field_table(model, tables),
                  "", f"## Disagreements with the consensus by sentence type, {model}", "", *audit_table(model, tables)]
    lines += ["", *FIGURE_TEXT, "",
              "The notes are synthetic, so these numbers show how the pipeline works and how to read",
              "agreement. They are not a finding about scouts or scouting reports.", ""]
    path = out_dir / "results.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
