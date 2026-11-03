# CC4D answer-type experiment: Gemini 3.5 Flash-Lite, full run (3 November 2026)

First model of the CC4D answer-type experiment (`docs/cc4d_answer_type_proposal.md`). Question
set `cc4d_answer_types_v1`; 41 runs (28 groups × the batch sizes in the proposal), forward and
reversed order, 25,711 calls, 102,753 scored fields.

* Model `gemini-3.5-flash-lite` on Vertex Express, 480p clips at 2 fps, temperature 0, output cap 4096, concurrency 8.
* Prompts: `experiments/cc4d_answer_types/prompts/` (user prompt v2, bare JSON placeholders; the multi_choice `not_visible` wording was added afterwards and is **not** in this run, see below).
* Judge for `open_text`: `gemini-3.5-flash-lite`, text only (`open_text_v1`), cached in each run's `evaluations/judge_cache.jsonl`. Scorer `typed_v2`.
* Run on the Azure VM, finished 3 November 13:30 UTC. Cost **US$433.71** from the pricing table (`cost_status: lower_bound`), not from the bill.

## Contents

* `runs/<run>/predictions.jsonl`, `run.json`: model answers and run metadata.
* `runs/<run>/evaluations/<id>/`: `field_scores.jsonl` (one row per field), `metrics.json`, `evaluation.json`.
* `summary.json`, `summarize.py`: per-run table. `python summarize.py` rebuilds it from `runs/`.
* Not saved (kept on the VM, `~/cc4d_at/runs_full_fl/`): raw responses, per-call records, the clip snapshot and the 36 MB smoke-test logs.

## Results

Field accuracy over all runs (the groups have different targets, so compare within a group):

| Answer type | Fields | Accuracy |
| --- | --- | --- |
| bool | 30,019 | 49% |
| single_choice | 30,019 | 36% |
| int | 6,348 | 47% |
| multi_choice | 6,348 | 44% |
| open_text | 6,220 | 43% |
| seconds | 23,799 | 12% |

Flash-Lite is close to chance on most cells, so effects of batch size and content are hard to see
with it alone. Base group: 38% at k = 1, 39% at k = 4, 39% at k = 8; reversed order 39% and 38%.

Not usable fields: 509 invalid answers, 607 unusable responses, 7 missing, 17 not executed.

## Known issues

* 2 calls failed after 3 attempts (retryable error): one in `fl_e5_loc_base_rev_k16` (16 fields) and one in `fl_e5_dur_base_k1` (1 field). These 17 fields count as `not_executed`.
* The `matched: ["none"]` judge reply (no reference mistake) made the first scoring attempt fail. `parse_verdict` now ignores `matched` when the reference is empty; all runs were scored with the fixed code.
* 295 of 6,348 multi_choice answers (4.6%) are invalid; 135 of them mention `not_visible`, the case the later prompt fix targets. That fix came after this run started, so they are scored as invalid here.
* The judge has not been checked against human labels (`scripts/judge_agreement.py`).
