# No-video control for the HD-EPIC baseline (2026-10-22)

The 2026-10-20 baseline (`results/hd_epic_q40_v1_gemini35_flash_lite_20261020/`) was re-run
for E1 (1 question per call), E3 (8) and E5 (32) with the video removed. Everything else
is the frozen request as sent on 2026-10-20:

- the same system and user prompts, batches, question order, model and generation settings;
- the same Vertex Express endpoint and adapter code;
- no video part, so the line naming the video clip is not sent either.

The prompts still say "about the video". Answers that are not a valid letter count as wrong;
there were none.

All 3,577 calls succeeded on 2026-10-22 between 03:45 and 03:56 UTC, with model version
`gemini-3.5-flash-lite`. They used 2.38M input and 0.11M output tokens, about US$0.99.

## Results

Accuracy is video-weighted. Intervals are 95% intervals from 2,000 bootstrap resamples of
videos (seed 20261021).

**Localization, the same 2,336 targets in every condition (73 videos)**

| Questions per call | With video | No video | Gap | 95% interval |
|---|---|---|---|---|
| 1 | 33.0% | 22.8% | +10.2 | +7.8 to +12.6 |
| 8 | 35.0% | 22.7% | +12.3 | +9.8 to +14.9 |
| 32 | 26.7% | 23.1% | +3.6 | +1.4 to +5.6 |

**One question per call, by question type (E1)**

| Type | Questions | Videos | With video | No video | Gap | 95% interval |
|---|---|---|---|---|---|---|
| Localization | 2,861 | 76 | 32.4% | 23.1% | +9.4 | +7.2 to +11.4 |
| How | 175 | 52 | 56.2% | 24.8% | +31.4 | +20.3 to +42.5 |
| Why | 160 | 60 | 69.2% | 28.6% | +40.6 | +27.6 to +52.2 |
| Weight | 16 | 10 | 18.3% | 0.0% | +18.3 | +0.0 to +40.0 |

Chance is 20% (five options). The with-video columns reproduce the baseline report exactly.

## Reading

- **Text-only accuracy stays flat** at about 23% from 1 to 32 questions per call. Larger
  batches do not leak answers between questions.
- **The drop at 32 questions per call is mostly lost video use.** The video's contribution
  falls from +12.3 points at 8 per call to +3.6 at 32.
- **How and why are answered from the video.** Their high with-video scores are not explained
  by the question text; without video they are close to chance.
- **The question text leaks little.** Text-only localization is about 3 points above chance.
- **Without video the model prefers early letters.** On the 2,336 E1 targets it picks A 26.5%
  and B 25.3%, against about 20% of targets each.

## Limits

- One model and one question order, as in the baseline.
- Run two days after the baseline, so a provider-side model update cannot be ruled out.
- Weight has 16 questions in 10 videos; do not interpret it.

## Files

- `responses/E{1,3,5}.jsonl.gz`: one line per call (call ID, raw text, token usage, finish reason).
- `per_field_no_video.csv.gz`: one row per question, paired with the baseline's with-video result.
- `summary.json`: the numbers above.

## Reproduce

```bash
set -a; source .env; set +a   # GEMINI_API_KEY = the Vertex Express key
PYTHONPATH=src python scripts/run_no_video_control.py                # dry run, no calls
PYTHONPATH=src python scripts/run_no_video_control.py --execute --max-cost-usd 2
PYTHONPATH=src python scripts/run_no_video_control.py --score
```

Outputs go to `outputs/no_video_control/` (git-ignored). The script only reads the baseline
folder and refuses to write under `results/`.
