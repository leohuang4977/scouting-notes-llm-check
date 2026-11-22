"""Run the whole check end to end.

Steps, in order:
  0. Record the git commit and whether the plan files changed since it.
  1. Rebuild the synthetic notes and the three human coders, and confirm
     they match the committed files in data/ (same seed, same data).
  2. Code every note with the keyword list.
  3. Code every note with the model through Ollama (or the MOCK coder).
  4. Score all coders on the 450 test notes, with bootstrap intervals,
     and apply the decision rule from PLAN.md.
  5. Draw the figure and write results.md.

Usage:
  python run_all.py --model qwen3:8b              (about 30 minutes)
  python run_all.py --model qwen3:8b gemma3:12b   (extra models: about 30 minutes or more each)
  python run_all.py --mock                        (no Ollama; writes to outputs_mock/)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from src.code_keywords import code_notes as code_with_keywords
from src.code_llm import DEFAULT_MODEL, MOCK_DIR_NAME, TooManyFailures, make_coder, run, safe_name
from src.codebook import DATA_DIR, REPO_ROOT, SEED, read_notes, read_sheet
from src.figure import draw
from src.generate_notes import generate_notes
from src.report import git_state, write_results
from src.score import score, write_tables
from src.simulate_coders import simulate


# ---------------------------------------------------------------- data

def check_or_write(table: pd.DataFrame, path: Path, rewrite: bool) -> None:
    """Write the table if the file is missing; otherwise confirm the file matches it.

    The committed data must be exactly what the generator makes from the
    seed. A mismatch usually means different package versions, so the run
    stops rather than scoring data nobody can reproduce.
    """
    expected = table.to_csv(index=False, lineterminator="\n")
    name = path.relative_to(REPO_ROOT).as_posix()
    if path.exists() and not rewrite:
        found = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        if found != expected:
            sys.exit(f"{name} does not match what the generator makes from seed {SEED}. "
                     "Check that the pinned versions in requirements.txt are installed, "
                     "or run again with --rewrite-data to replace the file.")
        print(f"{name} matches the generator (seed {SEED})")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(expected, encoding="utf-8", newline="\n")
    print(f"Wrote {name}")


# ---------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(description="Run the whole check end to end.")
    parser.add_argument("--model", nargs="+", default=[DEFAULT_MODEL], help="one or more Ollama model names")
    parser.add_argument("--mock", action="store_true", help="use the MOCK coder (no Ollama); writes to outputs_mock/")
    parser.add_argument("--rewrite-data", action="store_true", help="replace data/ files instead of checking them")
    args = parser.parse_args()

    # Read the git state before anything runs, so results.md reports the plan as it was at the start.
    git_info = git_state()
    print("Git: " + (f"last commit {git_info['commit'][:12]}" if git_info["commit"] else git_info["reason"]))

    print("Step 1 of 5: synthetic notes and human coders")
    check_or_write(generate_notes(), DATA_DIR / "notes.csv", args.rewrite_data)
    notes = read_notes()
    check_or_write(simulate(notes), DATA_DIR / "human_codes.csv", args.rewrite_data)
    human = read_sheet(DATA_DIR / "human_codes.csv")

    out_dir = REPO_ROOT / (MOCK_DIR_NAME if args.mock else "outputs")
    print("Step 2 of 5: keyword list")
    keyword = code_with_keywords(notes)
    out_dir.mkdir(parents=True, exist_ok=True)
    keyword.to_csv(out_dir / "keyword_codes.csv", index=False, lineterminator="\n")

    print("Step 3 of 5: model" + (" (MOCK coder, not a model)" if args.mock else ""))
    coders = [make_coder("MOCK", mock=True)] if args.mock else [make_coder(m, mock=False) for m in args.model]
    machine = {"keyword": keyword}
    manifests = []
    for coder in coders:
        cache_dir = out_dir / "cache" if args.mock else REPO_ROOT / "cache"
        try:
            machine[coder.label] = run(coder, notes, out_dir, cache_dir / f"{safe_name(coder.label)}.jsonl")
        except TooManyFailures as err:
            sys.exit(str(err))
        manifest_path = out_dir / f"manifest_{safe_name(coder.label)}.json"
        manifests.append(json.loads(manifest_path.read_text(encoding="utf-8")))

    print("Step 4 of 5: scoring on the test notes (1,000 bootstrap resamples)")
    tables = score(notes, human, machine)
    write_tables(tables, out_dir)

    print("Step 5 of 5: figure and summary")
    figure = draw(out_dir, [c.label for c in coders], mock=args.mock)
    results = write_results(out_dir, tables, manifests, git_info, mock=args.mock)
    print(f"Done. See {results.relative_to(REPO_ROOT).as_posix()} and {figure.relative_to(REPO_ROOT).as_posix()}")
    print("Prompt hash this run: " + manifests[0]["prompt_hash"])


if __name__ == "__main__":
    main()
