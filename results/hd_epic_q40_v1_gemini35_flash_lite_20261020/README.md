# HD-EPIC baseline results

**Dataset: `hd_epic_q40_v1`. Model: Gemini 3.5 Flash Lite. Completed: 20 October 2026.**

## Read the result

- [Three-page English report](report.pdf): Q1, Q2 and Q3, one page each.
- [Browser report](report.html) and [editable Markdown](report.md).
- [Technical appendix](technical_appendix.pdf): configuration, uncertainty, actual prompts/outputs and limitations.

## Analyze without a VM or API key

Run these commands from the repository root. Python 3.10+ is sufficient; no package installation is needed.

**1. Recompute Q1, Q2 and Q3 from the saved CSVs.**

```bash
python results/hd_epic_q40_v1_gemini35_flash_lite_20261020/analyze.py summary
```

**2. Find which exact questions become wrong at size 32.**

```bash
python results/hd_epic_q40_v1_gemini35_flash_lite_20261020/analyze.py flips \
  --from-size 8 --to-size 32 --output analysis/size8_to_size32.csv
```

The output contains all 2,336 paired targets. Filter `from_correct=True` and `to_correct=False` to inspect degradation. For 8 -> 32, 494 targets become wrong and 299 become correct; the net loss is 195 correct answers. These changes do not, by themselves, identify a mechanism.

**3. Read the exact input and output for a saved call.**

```bash
python results/hd_epic_q40_v1_gemini35_flash_lite_20261020/analyze.py trace \
  --experiment E5 --report-id hdepic-dense-001
```

For a question found in the paired CSV, use `--field-id loc_1234` in place of `--report-id`, with its actual ID. Run the trace for E3 and E5 to inspect its size-8 and size-32 prompts and answers. The trace includes offline ground truth separately from the model-facing request.

## Choose the right file

- [Per-field results](data/per_field.csv): **12,940 rows**. Use for accuracy, error analysis and paired comparisons. `predicted`, `target`, `correct`, `category`, `report_id`, `field_id` identify each result.
- [Per-call results](data/per_call.csv): **4,355 rows**. Use for token, cost and latency analysis. Costs/tokens cover the whole call, not each individual field.
- [Per-video cost and time for 32 questions](data/per_video_32_question_cost_time.csv): **365 rows = 73 videos x 5 sizes**. Each row sums all calls required to answer the same 32 targets. [Aggregated Q1 table](data/matched_localization.csv).
- [Question-family table](data/question_types.csv) and [summary.json](summary.json): Q2's primary metric is `video_weighted_accuracy`; `pooled_accuracy` is secondary. `summary.json` also contains Q3 common-target outcomes, paired changes and confidence intervals.
- [Actual inputs and outputs](data/all_model_inputs_outputs.jsonl.gz), [logs](logs/), [audit checks](data/audit.json) and [frozen selection](data/selection_manifest.json): use for tracing and reproducibility. Large text artifacts are gzip-compressed, losslessly.

## Avoid invalid comparisons

**Q1:** filter `matched_localization=True`, then compare E1/E2/E3/E4/E5 using `(report_id, field_id)`. Each setting scores the same 2,336 targets. Do not compare all 3,212 E1 questions with localization-only batches.

**Q2:** use all E1 rows. Within each subtype, first compute each video's accuracy, then average the videos equally. The four primary values are localization 32.42%, how 56.19%, why 69.19%, weight 18.33%. Weight has only 16 questions from 10 videos. The earlier pooled percentages are not the primary estimator specified by the proposal.

**Q3:** filter `e6_common_target=True` and compare E6_homogeneous with E6_mixed by `(report_id, field_id)`. Only 96 common localization targets are comparable; the other companion questions differ. Pure gets 41/96, mixed 40/96. Five targets change correct -> wrong; four change wrong -> correct.

**Timing:** `api_round_trip_s` includes upload, network and provider time; it excludes local preprocessing and is not backend-only model inference. For equal-work speed comparisons, use the summed API seconds per 32-target report, not mean latency per call.

**Scope:** results are from one model and one frozen order on the prepared question pool. All calls succeeded; legal JSON does not prove semantic grounding. Wrong multiple-choice answers are not a verified hallucination metric. The effective cost is US$43.4853869, excluding earlier superseded calls, smoke tests and VM charges.

## Rebuild the report and figures

This optional step needs Python 3.11+ and plotting/PDF packages. The report was rebuilt with Python 3.12.14. It makes **no API calls**.

```bash
python -m pip install -r results/hd_epic_q40_v1_gemini35_flash_lite_20261020/requirements-analysis.txt
python results/hd_epic_q40_v1_gemini35_flash_lite_20261020/manager_report.py
```

All paths are relative to this results directory. Rendering overwrites only generated report/figure/summary files here; predictions are unchanged. The scripts and original source records preserve technical provenance. See [artifact_manifest.json](artifact_manifest.json) for file checksums and exclusions.

## What is not committed

No API keys, `.env`, original videos, video base64 payloads, or live credential/session files are included. Original pipeline run directories and source archives remain in the VM/local evidence store; the repository contains analysis-ready records, exact saved prompt/response text, compressed logs, frozen inputs and audit receipts. The binary evidence ZIP is not duplicated in Git.
