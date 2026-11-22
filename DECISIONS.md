# Decisions

A dated log of design choices and of every change to the design after it was first written down. All entries are from before any model run.

## 2026-11-22, first build

**One seed, separate streams.** The project seed is 20261122. Each step draws from its own numpy stream of that seed (`default_rng([20261122, step])`): notes, human coders, the MOCK coder and the bootstrap. Changing how one step uses randomness then never shifts another step's draws.

**Codes first, text second.** The generator draws a note's true codes, then writes text that expresses exactly those codes. Overall tone is drawn first and the other codes are drawn conditional on it, so content and tone fit together. A check then makes sure the tone is defensible under the codebook rule: a mixed note has at least one favorable and one unfavorable point, and a positive or negative note with no closing judgment has no point on the other side. Draws that fail the check, or that come out shorter than 2 or longer than 5 sentences, are redrawn.

**Ambiguous share is decided before redrawing.** Whether a note is ambiguous is drawn once, at 15%, before the redraw loop, so redrawing for length cannot change the share.

**Unweighted kappa everywhere.** Three fields are ordinal, but each also has an off-scale code (not mentioned, not stated). Weighted kappa needs a distance between every pair of codes, and there is no natural distance between "plus" and "not mentioned". Unweighted kappa treats every disagreement alike.

**AC1 and PABAK use the codebook's number of codes.** Using the number of codes that happen to appear would let these statistics change across bootstrap resamples and coders for reasons unrelated to agreement.

**Decision rule on point estimates.** The intervals are reported but not used in the rule. Ties count as meeting a threshold, with a small tolerance for floating-point rounding.

**Notes the model fails to code.** They are left out of that model's agreement statistics, counted in the `n` column, and listed in the manifest. They are never filled in.

**Ollama details.**
- `think` is sent only to models that report the "thinking" capability through `/api/show`, because current Ollama returns an error when `think` is sent to a model without a thinking mode.
- The model digest comes from `/api/tags`, which lists digests. `/api/show` does not.
- The Ollama version comes from `GET /api/version`.
- `num_ctx` is set to 4096. The prompt is about 1,000 tokens plus the note. The manifest warns if a prompt comes within 512 tokens of the limit.
- Four attempts per note, waiting 2, 4 and 8 seconds. A 4xx error (for example a model that is not pulled) stops the run, since retrying cannot fix it. At temperature 0 with a fixed seed, a retry after an invalid answer will often return the same answer, so such notes usually end up reported as failed. With schema-constrained output they should be rare.

**Cache keyed by configuration.** Each cached answer carries a config id built from the model name, digest, prompt hash, options and think setting. Only answers with the current config id are reused, so a changed prompt or a re-pulled model starts fresh instead of reusing old answers.

**MOCK coder.** True codes, each changed to a random other code 8% of the time (20% on ambiguous notes). It is labelled MOCK in the code sheet, manifest, decision table, figure and results page, and `run()` refuses to write MOCK output anywhere but `outputs_mock/`, which git ignores.

**Modules.** Besides the pipeline steps there are `src/codebook.py` (codebook loader, seed, split rule, sentence types and paths, kept in one place so they cannot drift), `src/phrases.py` (the phrase banks, kept apart from the generator logic), `src/metrics.py` (the four statistics as plain functions, which the tests check directly) and `src/report.py` (the git check and the results page).

**Extra output.** `decision_by_field.csv` holds the decision rule's inputs and verdict per field, so the verdicts in `results.md` can be checked.

**Prompt not tuned.** No model could be run while building, so the prompt was written from the codebook and not tuned on dev notes. Its hash is in PLAN.md.

**Data check on every run.** `run_all.py` rebuilds the notes and human codes from the seed and compares them with the committed files (ignoring line-ending differences). A mismatch stops the run, since it usually means different package versions.

**No names.** Notes describe players by position and size only, so no note can match a real person. The surgery phrase avoids the procedure named after a player.

**Figure.** Drawn with a bare matplotlib `Figure` rather than pyplot, so no window or GUI backend is involved on Windows. Colors follow the entity across both panels: blue for the model, aqua for human coders, orange for the keyword list.

**Windows run command.** On Windows, `py -3.12 run_all.py` would start the system Python, not the `.venv` where the requirements are installed. The README uses `py -3.12` only to create the venv and `.venv\Scripts\python` for everything after.

**Tested environment.** Python 3.12 on Linux with the pinned versions in `requirements.txt`. Windows wheels for those exact versions exist for Python 3.12. The pipeline has not yet been run on Windows or with a real model.

## 2026-11-22, changes after an independent review (still before any model run)

**Keyword list pruned to dev phrases, and a test enforces it.** The review found five phrases that matched no dev note: "closed games", "late-inning", "lineup over", "no (real )?concerns" and "room to add". They could only have come from the phrase banks, so all five were removed. After the data were regenerated with more held-out wording, two more matched no dev note, "velocity is short" and "would not be shocked", and were removed too. ("late-inning" happens to appear in one regenerated dev note, but it stays out because it was not found by reading dev notes.) `tests/test_keywords.py` now checks that every top-level alternative in every keyword pattern matches at least one dev note.

**More held-out wording, flagged per note.** Every phrase list in `src/phrases.py` now has an explicit `_new` partner that appears only in test notes, including a held-out variant of every hard sentence. Before, only the last plain sentence of each list was held out and hard sentences never were. `data/notes.csv` gains `heldout_fields` (which fields used held-out wording) and `has_closing`. About 40% of the field sentences in test notes use held-out wording, and 401 of 450 test notes contain at least one.

**Keyword versus model is decided on held-out wording.** On wording it has seen, a keyword list written by the author of the templates is close to perfect by construction. So every statistic is also reported on the held-out-wording test notes for each field, and the "keyword list preferred" step uses that comparison. The keyword list's other clause (mean kappa with each coder) is compared with the human coders' agreement on the same held-out notes.

**Like-for-like human ceiling.** The ceiling clause used to compare a machine's kappa with the consensus against the humans' kappa with each other. A majority of three is less noisy than one coder, so that comparison flattered the machine. The clause now compares the machine's mean kappa with each individual human coder against the humans' mean kappa with each other. Kappa and AC1 with the consensus stay as the first clause. The clause is stated one-sided ("no more than .10 below"), which is how the code always worked. The figure's left panel and the summary line now show the like-for-like numbers.

**Verdict steps written out in full.** PLAN.md now lists all five steps, including "keyword list if it alone meets the bar", which the code had and the plan did not state, and a new first step, "codebook needs work", for any field where the human coders' mean kappa with each other is below .61.

**Slips tied to the sentence.** The simulated coders' slip rate now depends on the type of sentence that carries the field: 3% on plain wording seen in dev, 7% on held-out wording, 30% on hard sentences, scaled by field (role .8, velocity or power 1.0, command or approach 1.2, makeup 1.0, health .6, projection 1.6, tone 1.4) and by 1.5 for tone with no closing judgment. coder_c's random noise went from 5% to 3%. The human ceilings now land between about .74 and .88 on test notes. Before, slips depended on the text only through the ambiguous flag, so simulated people misread clean sentences that a regex read correctly.

**A shared, planted habit.** All three coders now miss hedged health mentions, and their misses are correlated. For each hedged mention, a shared draw decides whether it is easy to miss (40%). If it is, coder_a and coder_c miss it 75% of the time and coder_b 85%. Otherwise they miss it 5%, 25% and 5% of the time. coder_b's own habit (missing more of these than the others) is kept inside this. Realized: the consensus codes "no" on 26 of the 50 hedged mentions (52%), and on 24 of the 38 in the test split (63%). This gives the disagreement audit one sentence type where the consensus is wrong by design. It is declared in PLAN.md and in a comment in the codebook that is not part of the prompt.

**Audit by sentence type.** The disagreement audit is now broken down by plain wording, held-out wording, hard sentence and hedged health mention, as well as by field. The figure's right panel shows it by sentence type.

**Neither-right share computed directly.** The share where neither the coder nor the consensus is right is now counted directly instead of as one minus the other two shares, which could print as -0.0.

**The plan commit is checkable.** The README now has a `git init` and first commit before any model command. `run_all.py` records the last commit in `results.md`, says whether `PLAN.md`, `src/codebook.yaml` and `src/code_llm.py` changed since it, and checks the prompt hash against the hash in PLAN.md at that commit. With no repository it says plainly that the order cannot be shown.

**Speed check kept out of outputs/.** `python -m src.code_llm`, run on its own, now writes to `outputs_speedcheck/` (ignored by git), never `outputs/`. Only `run_all.py` writes to `outputs/`. Answers still go to the shared cache, so the full run reuses them.

**A broken server stops the run.** If 5 notes in a row fail, the run stops with the last error instead of retrying every remaining note.

**Realized shares after regeneration.** Ambiguous notes: 90 of 600 (15.0%). Typos: 32 of 600 (5.3%). Hedged health mentions: 50 of 600.

**Wording.** "Realistic" no longer describes the simulated slips. The README says that Claude wrote the templates, the keyword list and the simulated coders, and chose the slip rates.

## 2026-11-22, after the second review: the model must also pass on held-out wording
Verdict step 3 now requires the model to meet the bar on all test notes and on held-out-wording test notes. Before this, the model was judged on all test notes while the keyword list was judged on held-out wording, so a model that echoed familiar phrasing could be called usable where the keyword list failed on new wording. Changed before the first commit, in PLAN.md, README.md, src/score.py and tests/test_decision_rule.py.
