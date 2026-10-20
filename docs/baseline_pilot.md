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
