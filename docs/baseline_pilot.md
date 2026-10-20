# HD-EPIC baseline preparation and pilot

This workflow implements the October 17 proposal. The user has authorized only
offline preparation and the two-report pilot, not the full baseline execution.

Run on the Azure VM from `/home/azureuser/Capstone-14` using `.venv/bin/python`.
Credentials stay in the ignored `.env`; scripts never print them.

```bash
.venv/bin/python scripts/prepare_hd_epic_baseline.py \
  --output outputs/hd_epic_baseline_20261017 \
  --video-root /home/azureuser/datasets/hd-epic/Videos \
  --seed 20261017 --probe-videos

# Freeze and validate the pilot without inference:
.venv/bin/python scripts/run_hd_epic_baseline_pilot.py \
  --freeze outputs/hd_epic_baseline_20261017

# Execute or resume ONLY reports hdepic-dense-014 and hdepic-dense-022:
.venv/bin/python scripts/run_hd_epic_baseline_pilot.py \
  --freeze outputs/hd_epic_baseline_20261017 --execute

.venv/bin/python scripts/report_hd_epic_baseline_pilot.py \
  --freeze outputs/hd_epic_baseline_20261017
```

Preparation refuses to overwrite an existing freeze. Use a new output directory
for a different selection or prompt. The pilot runner checks the frozen source
hashes, snapshots and rendered inputs before execution. It resumes successful
calls, uses one attempt per call and does not automatically retry failures.
Its default US$1.50 preflight estimate limit is not a provider billing cap.

`method: {name: batched, batch_size: 8}` splits a report's fixed field order into
consecutive independent calls. It never combines videos or drops a final partial
batch. E2–E5 use exactly 32 fields, so all batches have the requested size.

The full freeze contains 4,355 nominal calls: E1 3,212; E2 584; E3 292; E4 146;
E5 73; E6 homogeneous/mixed 24 each. The pilot contains 124 calls and runs E5
first to check 32-answer completeness. No previous smoke result is reused, since
the approved system prompt changed.

E6 selects four common targets without reading target labels: rank the seeded
localization order by surface cues against the fixed how/why companions, then
take four targets and four other localization companions. Both conditions keep
the targets at positions 1–4. Surface flags and the complete question/option
review are retained; this does not establish absence of semantic answer leakage.

Use only shared localization targets for the Q1 size comparison, and only the
four common targets per report for Q3. For Q2, weight eligible videos equally
within each subtype and show question counts. Reports 014/022 have no weight
questions, so the pilot cannot validate inference on that subtype.

Outputs include source hashes, source-code fingerprints, seed, selection IDs,
plans, rendered requests, video inventory, exclusions, raw API responses,
checkpoints, token usage, cost, latency, paired scores and full logs. Base64
request bodies and prepared media remain on the VM and are excluded from the
compact evidence ZIP. All generated data lives under ignored `outputs/`; the
source benchmark is unchanged.

## Integrity fixes and experiment identification

`verify_derived_ground_truth` compares every selected target and target text with
`data/hd_epic_q40_v1/ground_truth.jsonl` before inference. The derived file's hash
is saved in `pilot_plan.json` and checked again before scoring. Intentional
alternative ground truth remains possible through the separate offline evaluator.

New prepared clips live in `pilot/media/verified-v1/`; legacy media and snapshots
are left intact. Clips are written to a temporary file, fully decoded, checked
against the requested duration, and published with an atomic rename and SHA-256
receipt. Reuse checks the receipt/hash each time and decodes on first use in a
process. Missing receipts, partial files, decode failures and changed hashes cause
reconstruction from the original video. An invalid reconstruction stops before HTTP.

Snapshot verification hashes the saved config representation, so old configs
without `batch_size` and newer configs with explicit `batch_size: null` both
verify against their original fingerprints. Resume compares current normalized
configs only after that integrity check. No historical file or fingerprint is
rewritten. Reviewed package upgrades still require `--allow-code-change`; this
permits code changes, never changed frozen model inputs.

Cross-run result import/deduplication is deliberately not implemented. Identify
results by experiment/run directory, call ID, requested field IDs, and the frozen
method/batch-size configuration. The same run already resumes successful calls
without sending them again. A newly created experiment is a separate execution;
choose and combine its results explicitly during analysis.

## Origin of overlapping-event questions

The two `put carrot` questions already exist as distinct entries
`fine_grained_action_localization_6239` and `_4267` in the
[official VQA JSON](https://github.com/hd-epic/hd-epic-annotations/blob/9a47a428a90d83a2fd5d87356afe97bef5649989/vqa-benchmark/fine_grained_action_localization.json).
Their original input windows are 04:32.300–09:41.860 and
07:08.089–13:34.440 on `P01-20240203-130505`. Both correct choices refer to
08:39.490–08:39.730, with different distractors and answer indices (D/E).

`build_hd_epic_dense.py` reads these existing records, preserves choices and
`correct_idx`, and converts timestamps. It does not generate new questions.
It collects both records into one larger report without event deduplication;
that assembly exposes cross-question hints absent from independent calls. The
source records do not establish whether the benchmark creators intentionally
sampled the same event twice or which annotation-generation step produced it.
The original data and the frozen question selection are unchanged by this fix.
