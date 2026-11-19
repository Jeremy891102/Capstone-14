# CC4D answer-type experiment: Gemini 3.5 Flash-Lite, full run (3 November 2026)

First model of the CC4D answer-type experiment (`docs/cc4d_answer_type_proposal.md`). Question
set `cc4d_answer_types_v1`; 41 runs (28 groups × the batch sizes in the proposal), forward and
reversed order, 25,711 calls, 102,753 scored fields.

* Model `gemini-3.5-flash-lite` on Vertex Express, 480p clips at 2 fps, temperature 0, output cap 4096, concurrency 8.
* Prompts: `experiments/cc4d_answer_types/prompts/` (user prompt v2, bare JSON placeholders; the multi_choice `not_visible` wording was added afterwards and is **not** in this run, see below).
* Judge for `open_text`: `gemini-3.5-flash-lite`, text only (`open_text_v1`), cached in each run's `evaluations/judge_cache.jsonl`. Scorer `typed_v2`.
* Run on the Azure VM, finished 3 November 13:30 UTC. Cost **US$433.71** from the pricing table (`cost_status: lower_bound`), not from the bill.

## Contents

* `runs/<run>/predictions.jsonl`, `run.json`: model answers and run metadata (first pass, 3 Nov).
* `runs_rerun/rr_<run>/`: second pass of the reports that have a `seconds` or `multi_choice` field (4 Nov, see "Rerun"). `merged.py` gives the final scores: a rerun report replaces the same report in `runs/`.
* `runs/<run>/evaluations/<id>/`: `field_scores.jsonl` (one row per field), `metrics.json`, `evaluation.json`.
* `summary.json`, `summarize.py`: per-run table from the merged scores. `analysis.md` (Chinese) / `analysis.en.md`, `figs/`, `make_figs.py`, `make_q234.py`: findings per research question (`make_q234.py` does the chance / majority correction for Q2–Q4).
* Not saved (kept on the VM, `~/cc4d_at/runs_full_fl/`): raw responses, per-call records, the clip snapshot and the 36 MB smoke-test logs.

## Results

Field accuracy over all runs (the groups have different targets, so compare within a group):

| Answer type | Fields | Accuracy |
| --- | --- | --- |
| bool | 30,019 | 49% |
| single_choice | 30,019 | 36% |
| int | 6,348 | 47% |
| multi_choice | 6,348 | 45% |
| open_text | 6,220 | 43% |
| seconds | 23,799 | 23% |

Flash-Lite is close to chance on most cells, so effects of batch size and content are hard to see
with it alone. Base group: 40% at k = 1, 41% at k = 4, 41% at k = 8; reversed order 41% and 40%. (These are the final `typed_v3` numbers, including the rerun; the table above was computed before the rerun and the type rows are updated here.)

Not usable fields: 434 invalid answers, 346 unusable responses, 7 missing, 17 not executed.

## Known issues

* 2 calls failed after 3 attempts (retryable error): one in `fl_e5_loc_base_rev_k16` (16 fields) and one in `fl_e5_dur_base_k1` (1 field). These 17 fields count as `not_executed`.
* The `matched: ["none"]` judge reply (no reference mistake) made the first scoring attempt fail. `parse_verdict` now ignores `matched` when the reference is empty; all runs were scored with the fixed code.
* multi_choice: the `not_visible` prompt fix was added after the first pass and did not help (see Rerun).
* The open_text judge agrees with 100 hand labels 78% of the time (kappa 0.49, `labels/`); it is stricter than the labeller (19 of 22 disagreements are judge-incorrect, human-correct), so err_open_text is probably underestimated. The judge is kept as is.

## Rerun and rescoring (4 November)

After the first pass, two prompt changes were rerun on every report with a `seconds` or `multi_choice` field: 9,633 calls, all succeeded, US$171.35 (first pass US$433.71).

* `seconds` hint "number of seconds, as a plain number (125, not 2:05 or 0205)" (commit 1b2a8ab); `multi_choice` `not_visible` wording (commit 6b75bbe, missing in the first pass).
* **The prompt changes did not help.** The model still wrote times as MMSS (target 596 s, answer 1004), and `multi_choice` answers are often option names (`["Timing"]`) instead of letters.
* So both runs were rescored with scorer `typed_v3` (`docs/scoring_rules.md`): for timestamp fields an integer such as 1004 is also read as 10:04 (2,515 fields were correct only this way), and a `multi_choice` item that is an option's text counts as its letter. Both rules were set after seeing the results. Open-text scoring and the judge are unchanged.
* Effect: loc seconds 4.5% → about 25% (flat across question counts); err_multi invalid at k = 1 23 → 18; other cells move 1 point or less. Overall accuracy at k = 1 / 4 / 8: 40.1% / 41.3% / 41.2%.
* The strict first scores (`typed_v2`) are kept in `summary_typed_v2.json` (that file already has the rerun merged in); the evaluation folders hold `typed_v3` only. 71 first-pass and 30 rerun calls still fail on leading-zero numbers (`0954`) and count as invalid.
