# Scouting notes LLM check

Can a small language model running locally code short scouting notes as reliably as human coders? And when the model and the human coders disagree, who is wrong?

This repository runs that check on 600 synthetic notes about invented amateur players. It is a rebuilt, general version of a pipeline I wrote for my dissertation, run on synthetic data because participant data cannot leave the lab.

## What it does

1. Generates 600 short scouting notes (2 to 5 sentences each) from a seeded generator. Each note is written from a set of true codes, so the right answer for every field is known. Some wording is held out of the first 150 notes (the dev split) and appears only in the other 450 (the test split), and each note records which fields used it.
2. Simulates three human coders and forms their consensus. Their slips and habits are set in the simulation, including one habit all three share on purpose (see below).
3. Codes every note with a keyword list written from the dev notes only.
4. Codes every note with a local model (qwen3:8b through Ollama), one call per note, with the answer constrained to the codebook.
5. Scores everyone on the 450 test notes: agreement with the consensus (percent agreement, Cohen's kappa, Gwet's AC1, PABAK), agreement with each individual human coder, accuracy against the true codes, and bootstrap 95% intervals. Every number is reported on all test notes and on the test notes with held-out wording.
6. Applies a decision rule written in [PLAN.md](PLAN.md) and committed before any model run: on which fields would I trust the model, on which is a keyword list enough, where are people still needed, and where does the codebook itself need work?

## Why synthetic data

The notes are synthetic for two reasons. The original data cannot leave the lab. And with synthetic notes the true code of every field is known, which makes it possible to check whether the model or the human consensus is wrong when they disagree. Real data cannot answer that.

The notes vary in some of the ways real ones do: scout jargon ("flashes plus raw", "plus-plus", "fringy"), hedges, negations ("no concerns with his strike-throwing"), missing information, typos in about 5% of notes, and about 15% of notes written to be hard on purpose. No real players, scouts, teams, schools or text appear.

## How to run it on Windows

You need Python 3.12, Git and [Ollama](https://ollama.com). Open PowerShell in the repository folder.

**1. Set up and check the code without a model.**

```
py -3.12 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python -m pytest
.venv\Scripts\python run_all.py --mock
```

The tests take under a minute. One test compares kappa with scikit-learn and is skipped if scikit-learn is not installed. The `--mock` run replaces the model with a MOCK coder (true codes plus noise) and writes only to `outputs_mock/`, which git ignores.

**2. Commit the plan and code before any model command.**

```
git init
git add -A
git commit -m "Plan and code before any model run"
```

This commit is what makes "written down before the model ran" checkable. Every later run records this commit in `outputs/results.md`, and says whether `PLAN.md`, the codebook or the prompt code changed since. The repository will be pushed to github.com/leohuang4977/scouting-notes-llm-check after this commit.

**3. Run the model.**

```
ollama pull qwen3:8b
.venv\Scripts\python -m src.code_llm --model qwen3:8b --limit 10
.venv\Scripts\python run_all.py --model qwen3:8b
```

The second line is a speed check on 10 notes. It writes to `outputs_speedcheck/` (ignored by git), never to `outputs/`, and the full run reuses its answers. The full run makes 600 calls. At about 3 seconds per call with a GPU it takes about 30 minutes. On a CPU only, allow one to two hours or more. Progress prints every 50 notes. If the run stops, run the same command again and it picks up from the cache. If 5 notes in a row fail, it stops and says why.

**4. Commit the results** as a second commit: `git add -A` then `git commit -m "Results from qwen3:8b"`.

Other useful commands:

| Command | What it does |
|---|---|
| `.venv\Scripts\python -m src.code_llm --show-prompt` | Prints the exact prompt the model sees and its hash |
| `.venv\Scripts\python run_all.py --model qwen3:8b llama3.1:8b gemma3:12b` | Adds more models (pull each one first). Each adds roughly another half hour or more |

On macOS or Linux, use `python3.12 -m venv .venv` and `.venv/bin/` in place of `.venv\Scripts\`.

## What comes out

Everything lands in `outputs/`.

| File | What it is |
|---|---|
| `results.md` | The git check (commit, and whether the plan files changed since), run details, a five-line summary, and tables by field and by sentence type |
| `agreement_by_field.csv` | For all test notes and for held-out wording: every coder against the consensus (percent agreement, kappa, AC1, PABAK, accuracy against the truth), the keyword list and the model against each human coder, the human coders against each other, and the consensus against the truth. Each number has a 95% interval |
| `disagreement_audit.csv` | Where the model (or the keyword list) differs from the consensus: how often it was right, how often the consensus was right, how often neither. By field and by sentence type |
| `decision_by_field.csv` | The decision rule applied field by field, with its inputs |
| `figures/agreement_by_field.png` | Left: per field, the keyword list's and the model's mean kappa with each human coder, beside the human coders' mean kappa with each other. Right: when the model and the consensus disagree, who was right, by sentence type |
| `model_codes_<model>.csv`, `keyword_codes.csv` | Every code each coder gave (for example `model_codes_qwen3_8b.csv`) |
| `manifest_<model>.json` | What is needed to reproduce the model run: model digest, Ollama version, options, prompt hash, timings, failed notes |

The synthetic notes and the human codes are committed in `data/`. Each run rebuilds them from the seed and stops if they do not match.

## How the code is laid out

| File | Role |
|---|---|
| `run_all.py` | Runs every step in order |
| `src/codebook.yaml` | The seven fields, their codes, definitions and rules. The model prompt is built from this file |
| `src/codebook.py` | Loads the codebook. Holds the seed, the dev/test split rule, sentence types and file paths |
| `src/phrases.py` | The phrase banks the notes are written from, with the held-out wording |
| `src/generate_notes.py` | Draws true codes, then writes a note that expresses them |
| `src/simulate_coders.py` | Three simulated human coders and their consensus |
| `src/code_keywords.py` | The keyword baseline |
| `src/code_llm.py` | The Ollama client: schema-constrained JSON, thinking off, temperature 0, fixed seed, retries, resumable cache, manifest. Also the MOCK coder |
| `src/metrics.py` | Percent agreement, Cohen's kappa, Gwet's AC1, PABAK |
| `src/score.py` | Bootstrap intervals, the disagreement audit and the decision rule |
| `src/figure.py` | The two-panel figure |
| `src/report.py` | The git check and `results.md` |
| `tests/` | Checks that the generator is repeatable, that held-out wording never reaches dev notes, that every keyword phrase appears in a dev note, that the statistics match hand-worked values, that the decision rule works as written, that the splits do not overlap and that MOCK output is always labelled |

## The codebook

| Field | Codes |
|---|---|
| role | starter, reliever, position_player |
| velo_or_power | below_average, average, plus, not_mentioned |
| command_or_approach | concern, neutral, strength, not_mentioned |
| makeup | positive, negative, not_mentioned |
| health_flag | yes, no |
| projection | limited, some, high, not_stated |
| overall_tone | negative, mixed, positive |

The rules for hard cases are in `src/codebook.yaml`. For example, a pitcher's velocity is coded by where the fastball sits, not its peak, and "no concerns with his command" is neutral, not a strength. Some rule examples reuse wording from the notes, which helps every coder on those sentences.

## The decision rule

Written in [PLAN.md](PLAN.md) before any model run, and applied to the test notes field by field.

A coder meets the bar if its kappa and its AC1 with the consensus are both at least .61, and its mean kappa with each human coder is no more than .10 below the human coders' mean kappa with each other. The second clause compares like with like: one coder against one coder.

Each field then gets one verdict, in this order:

1. **Codebook needs work** if the human coders' mean kappa with each other is below .61.
2. **Keyword list** if, on test notes with held-out wording, the keyword list meets the bar and the model's kappa is not at least .05 higher.
3. **Model** if the model meets the bar on all test notes and also on held-out-wording test notes.
4. **Keyword list** if it meets the bar on held-out wording.
5. **Human coders** otherwise.

Kappa alone can mislead when one code is very common (most notes have no health flag), which is why AC1 has to agree.

## What this can and cannot show

The first question, whether a local model codes these notes well enough to meet the bar in PLAN.md, is answered here for synthetic notes.

The second question is a demonstration of the method. The simulated coders are built from the true codes with slips, so outside one planted habit their consensus is right about 97% of the time or more by design. When the model disagrees with it on ordinary sentences, the model will usually be the one that is wrong. To check that the audit can catch a consensus that is wrong, all three simulated coders share one habit, written into PLAN.md before the run: they often miss hedged health mentions, and they tend to miss the same ones. If the model follows the codebook rule, the audit should show it right on those notes. The answer key on hard sentences is one reading of the codebook, so a model that picks another defensible code is scored wrong.

The keyword list was written from the dev notes by the same author as the templates, so on familiar wording it is close to perfect. The comparison that matters is on test notes with wording the dev notes never use, and that comparison decides whether the keyword list is preferred.

With real notes there are no true codes, so a senior evaluator would read a sample of the disagreements and decide.

## Reading the results

Results from a qwen3:8b run are in [outputs/results.md](outputs/results.md), with the figure in `outputs/figures/`.

The notes are synthetic, so the numbers show how the method works and how I read agreement. They are not a finding about scouts or scouting reports. Three more limits matter:

- The notes come from templates. Real notes have far more varied wording. The same author, Claude, wrote the templates and the keyword list, so the keyword list's scores on familiar wording are a generous estimate.
- The human coders are simulated. Their slip rates and habits were chosen by Claude when building the sample, not measured.
- The model prompt was written from the codebook and not tuned, because no model was run while the code was being built.

## Using this with real notes

- **Write the codebook with the people who write the notes.** Every scouting department uses words like "projects" or "fringy" in its own way. The rules for hard cases matter more than the field list.
- **Measure people first.** Have two or three staff code a few hundred notes blind. Their agreement with each other is the ceiling, and any field where people cannot agree needs a better definition before a model sees it.
- **Split by time, not at random.** Tune on older seasons and test on the newest one, so the check also catches changes in how scouts write.
- **Write the decision rule down and commit it before running the model,** as in PLAN.md here.
- **Keep the model local.** With Ollama the notes never leave the organization's machines, and the manifest pins the exact model weights used.
- **Read the disagreements.** Real notes have no answer key, so a senior evaluator would read a sample of the notes where the model and the people disagree and say who is right. That replaces the true codes used here.
- **Recheck every season.** New scouts and new terms change the language, and agreement can drift.

## How this was built

Claude (Anthropic) wrote the pipeline and the synthetic-note generator. I rebuilt my dissertation pipeline on synthetic scouting notes, ran the model on my own machine, and committed the plan before the run.

## License

MIT. See [LICENSE](LICENSE).
