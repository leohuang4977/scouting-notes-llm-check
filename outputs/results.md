# Results

Run finished 2026-11-22 07:39 UTC. Test notes: 450. Bootstrap: 1,000 resamples (seed 20261122).

## Was the plan written down before this run?

- Last commit before this run: ae70ef1199b2fe3d2cd466946ec83cbffc250dd7 (2026-11-22 02:21:53 -0500).
- `PLAN.md` (the plan and decision rule): unchanged.
- `src/codebook.yaml` (the codebook the prompt is built from): unchanged.
- `src/code_llm.py` (the code that builds the prompt): unchanged.
- Prompt hash used in this run, e289bf7c5709, matches the hash in PLAN.md at that commit (e289bf7c5709).

## Coders

- qwen3:8b (MODEL). Digest: 500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41. Ollama: 0.35.1.
  Options: {"temperature": 0, "seed": 20261122, "num_ctx": 4096}. Thinking: think=false sent (model supports thinking; turned off).
  Notes coded: 600 of 600. Failed: 0. Median seconds per call: 1.22.

## Summary for qwen3:8b

1. Under the rule in PLAN.md, the model meets the bar on 3 of 7 fields: role, velocity or power, command or approach.
2. Verdict by field. The model: role, command or approach. Keyword list: velocity or power. Human coders still needed: makeup, health flag, projection, overall tone. Codebook needs work: none.
3. Mean kappa with each human coder, median across fields: the model .77, keyword list .69. The human coders with each other: .82.
4. On wording held out of the dev notes, median kappa with the consensus across fields: the model .81, keyword list .36 (on all test notes: .82 and .73).
5. Where the model and the consensus disagreed (527 codes), the model matched the true code 8% of the time and the consensus 91%. On hedged health mentions, where the simulated coders share a planted habit, the model was right in 82% of 22 disagreements.

## By field for qwen3:8b

| Field | Model kappa with consensus [95% CI] | Model AC1 with consensus | Model mean kappa with each coder | Human coders with each other | Keyword kappa, all / held-out | Model kappa, held-out | Verdict |
|---|---|---|---|---|---|---|---|
| Role | .95 [.92, .97] | .95 | .89 | .88 | .88 / .75 | .89 | model |
| Velocity or power | .81 [.77, .85] | .81 | .76 | .85 | .95 / .85 | .86 | keyword list |
| Command or approach | .89 [.86, .93] | .90 | .83 | .86 | .70 / .24 | .85 | model |
| Makeup | .85 [.80, .90] | .90 | .78 | .82 | .73 / .00 | .54 | human coders |
| Health flag | .82 [.75, .88] | .93 | .77 | .75 | .67 / -.09 | .60 | human coders |
| Projection | .33 [.29, .38] | .29 | .30 | .80 | .79 / .42 | .43 | human coders |
| Overall tone | .63 [.58, .68] | .63 | .57 | .74 | .63 / .36 | .81 | human coders |

## Disagreements with the consensus by sentence type, qwen3:8b

| Sentence type | Disagreements | Model right | Consensus right | Neither |
|---|---|---|---|---|
| Plain wording seen in dev | 373 | 2% | 97% | 1% |
| Wording held out of dev | 111 | 6% | 94% | 0% |
| Hard sentence (ambiguous note) | 21 | 38% | 57% | 5% |
| Hedged health mention | 22 | 82% | 18% | 0% |
| All sentences | 527 | 8% | 91% | 1% |

Figure (`figures/agreement_by_field.png`). Left: for each field, the keyword list's and the model's mean
Cohen's kappa with each of the three human coders, beside the human coders' mean kappa with each other,
with bootstrap 95% intervals. The grey tick marks the human value minus .10, the threshold in the
decision rule. Right: on the test notes where the model and the coders' consensus disagree, the share
where the model matched the true code, where the consensus did, and where neither did, by the type of
sentence the field was written in.

The notes are synthetic, so these numbers show how the pipeline works and how to read
agreement. They are not a finding about scouts or scouting reports.
