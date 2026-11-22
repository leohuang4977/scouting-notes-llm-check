# Plan

Written 2026-11-22 and committed to the repository before any model run. The README's first steps make that commit, and `results.md` records the commit and whether this file changed since. Results that do not follow this plan should be read as exploratory.

## Question

Can a small open-weight model running locally code short amateur scouting notes as reliably as human coders do? And when it disagrees with them, is the model or the coders wrong?

The notes are synthetic, so every note's true codes are known. That makes the second half answerable here, within the limits set out under "What the audit can and cannot show".

## Data

- 600 synthetic notes on invented amateur players (pitchers and position players), 2 to 5 sentences each, from a seeded generator (seed 20261122).
- The notes vary in wording: scout jargon ("flashes", "projects", "plus-plus", "fringy"), hedges, negations ("no concerns with his strike-throwing"), mixed signals, missing information and typos in about 5% of notes.
- About 15% of notes are flagged `ambiguous`. Each of these has at least one hard sentence that needs a codebook rule to resolve.
- No real players, scouts, teams, schools or text.

## Split and held-out wording

By note id. Dev = N0001 to N0150. Test = N0151 to N0600.

- Every phrase list has a held-out partner. Held-out sentences, including held-out hard sentences, appear only in test notes. `data/notes.csv` records for each note which fields used held-out wording (`heldout_fields`).
- The keyword list is written from dev notes only. A test checks that every phrase in it appears in at least one dev note.
- The model prompt is written from the codebook. It has not been tuned, because no model was run while building. If it is tuned on dev notes later, the new prompt hash must be written below and committed before the test run.
- All reported numbers come from the 450 test notes, scored once.

## Codebook

Full definitions and rules are in `src/codebook.yaml`. The model prompt is built from that file.

| Field | Type | Codes |
|---|---|---|
| role | nominal | starter, reliever, position_player |
| velo_or_power | ordinal | below_average, average, plus, not_mentioned |
| command_or_approach | ordinal | concern, neutral, strength, not_mentioned |
| makeup | nominal | positive, negative, not_mentioned |
| health_flag | binary | yes, no |
| projection | ordinal | limited, some, high, not_stated |
| overall_tone | ordinal | negative, mixed, positive |

## Coders

- **Three simulated human coders** (coder_a, coder_b, coder_c). They do not read the text. Each starts from the true code and slips at a rate set by the type of sentence that carries the field: low on plain wording seen in dev (3%), higher on held-out wording (7%) and much higher on hard sentences (30%), scaled up for projection and tone and for tone without a closing judgment. Each coder has one habit: coder_a leans positive on tone (mixed coded as positive about 15% of the time), coder_b misses hedged health mentions more often than the others, and coder_c adds plain random noise (3% of codes).
- **A shared habit, planted on purpose.** All three coders often miss hedged health mentions ("may have had some forearm tightness, not confirmed"), which the codebook says code `yes`. The misses are correlated: 40% of hedged mentions are easy to miss for everyone (each coder misses them 75 to 85% of the time), and the rest are missed 5 to 25% of the time. So the consensus is expected to be wrong on a large share of hedged mentions, and the disagreement audit has a known answer to recover there.
- **Consensus** = the majority code. A three-way split goes to the true code with probability .8, otherwise to coder_a's code.
- **Keyword list**: regular expressions written from dev notes.
- **Model**: qwen3:8b through the Ollama HTTP API, thinking off, temperature 0, seed 20261122, one call per note returning all seven fields as schema-constrained JSON. Optional extra models: llama3.1:8b, gemma3:12b.
- **MOCK coder** (true codes plus noise): exists only to test the pipeline. Its output is labelled MOCK and written only to `outputs_mock/`.

Prompt hash: `e289bf7c5709`

(Print the prompt and its hash with `python -m src.code_llm --show-prompt`. `run_all.py` reports whether the run used this prompt.)

## Metrics (test split)

- **Against the consensus**, per field, for each coder: percent agreement, Cohen's kappa (unweighted), Gwet's AC1, and PABAK = (k * po - 1) / (k - 1), where k is the number of codes for the field in the codebook.
- **Against each human coder**: for the keyword list and the model, kappa with each of the three human coders, averaged over the three.
- **Human ceiling**: the human coders' kappa with each other, averaged over the three pairs. This is compared with the previous line, like with like. It is not compared with kappa against the consensus, because a majority of three is less noisy than any one coder and so flatters whoever is compared with it.
- **Accuracy against the true codes** for the consensus, each human coder, the keyword list and the model.
- **Two sets of notes**: every statistic is reported on all test notes and on the held-out-wording test notes for that field (the test notes whose sentence for that field uses wording that never appears in dev notes).
- **Disagreement audit**: on the notes where a machine coder's code differs from the consensus, the share where the machine matches the truth, where the consensus does, and where neither does. Reported by field and by sentence type: plain wording seen in dev, held-out wording, hard sentence, hedged health mention.
- **Intervals**: 95% intervals from 1,000 bootstrap resamples of the test notes (seeded), percentile method.

## Decision rule

Applied to point estimates, field by field. An undefined statistic does not meet a threshold. Ties meet it.

**The bar.** A coder meets the bar on a set of notes if:
1. its kappa and its AC1 with the consensus are both at least .61, and
2. its mean kappa with each human coder is no more than .10 below the human coders' mean kappa with each other on the same notes.

**Verdict per field, in this order:**
1. **Codebook needs work** if the human coders' mean kappa with each other (all test notes) is below .61. People who cannot agree need a better definition before a machine is judged against them.
2. **Keyword list** if, on held-out-wording test notes, the keyword list meets the bar and the model's kappa with the consensus is not at least .05 higher than the keyword list's. The held-out comparison decides this step because on wording it has seen, a keyword list written by the author of the templates is close to perfect by construction.
3. **Model** if the model meets the bar on all test notes and also on held-out-wording test notes (so a model that only echoes familiar phrasing is not called usable).
4. **Keyword list** if it meets the bar on held-out-wording test notes.
5. **Human coders** otherwise.

## What the audit can and cannot show

- The simulated coders are built from the true codes, so outside the planted habit their consensus is right about 97% of the time or more by design. Where a model disagrees with it on ordinary sentences, the model will usually be the one that is wrong. That part of the audit mostly restates the model's own accuracy.
- On hedged health mentions the consensus is wrong by design. If the model follows the codebook rule, the audit should show it right on those notes. This tests whether the audit can catch a consensus that is wrong in a consistent way, which is the case that matters with real coders.
- The answer key on hard sentences is the generator author's reading of the codebook. A model that picks another defensible code is scored wrong.

## Outputs

- `outputs/agreement_by_field.csv`
- `outputs/disagreement_audit.csv`
- `outputs/decision_by_field.csv`
- `outputs/figures/agreement_by_field.png`. Left panel: per field, the keyword list's and the model's mean kappa with each human coder, beside the human coders' mean kappa with each other, with the rule threshold (human value minus .10). Right panel: of model-consensus disagreements, the share where the model was right, the consensus was right, or neither, by sentence type.
- `outputs/results.md`: the git check, run details, a five-line summary, and tables by field and by sentence type.

## Runtime

600 calls at about 3 seconds each (qwen3:8b, thinking off, with a GPU) is about 30 minutes. On a CPU only, allow one to two hours or more.
