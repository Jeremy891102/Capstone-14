# HD-EPIC Q2 extension: all how / why / weight questions on the 76 baseline videos

**Dataset: [`hd_epic_q2_full_v1`](../../data/hd_epic_q2_full_v1/). Model: Gemini 3.5 Flash Lite. Completed: 22 October 2026.**

The E1 baseline ([hd_epic_q40_v1 results](../hd_epic_q40_v1_gemini35_flash_lite_20261020/)) had few non-localization questions (how 175, why 160, weight 16). This run adds every remaining how / why / weight question from the official HD-EPIC VQA benchmark that falls on the same 76 videos (514 new questions), so no new videos were needed. Localization is unchanged (E1's 2,861 questions).

## Q2 result (primary)

Video-weighted accuracy: each video's accuracy within a subtype, then the mean over videos. 95% percentile interval from 10,000 video-bootstrap resamples (seed 20261020), as in the E1 report.

Subtype | Accuracy | 95% interval | Correct / Q | Videos | E1-only (previous)
--- | --- | --- | --- | --- | ---
Localization | 32.42% | 30.27-34.53% | 928/2861 | 76 | 32.42% (30.27-34.53%)
How | 50.24% | 43.74-56.65% | 216/421 | 67 | 56.19% (47.01-65.32%)
Why | 71.25% | 65.57-76.72% | 269/405 | 72 | 69.19% (60.53-77.22%)
Weight | 36.67% | 21.33-53.33% | 16/39 | 25 | 18.33% (0.00-40.00%)

- The E1 ordering holds: Why > How > Localization, with non-overlapping intervals. How drops from 56% to 50%, and its interval narrows from about ±9pp to about ±6.5pp.
- Weight is still only 39 questions from 25 videos. It moved from 18% to 37%, which shows how unstable the 16-question estimate was. Treat it as directional only.
- The E1-only intervals differ from the published ones by less than 0.4pp, because this script uses Python's `random` instead of numpy. Point estimates are identical.

## Window check

Subtype | Original q40_v1 windows (351 Q) | New windows (514 Q)
--- | --- | ---
How | 56.19% (46.92-65.43%) | 47.08% (38.57-55.49%)
Why | 69.19% (60.69-77.17%) | 68.31% (60.09-76.26%)
Weight | 18.33% (0.00-40.00%) | 50.00% (30.56-72.22%)

Each pair of intervals overlaps, so this run does not show a window effect. How is 9pp lower on new windows. Two things differ for those windows, besides the questions themselves:
- New windows are a little shorter on average (mean 7.6 vs 8.8 min, median 28K vs 32K input tokens). Each new window uses its video's q40_v1 window length, and the new questions come more from videos with short windows.
- A new window was placed at a random offset around one question, while q40_v1 windows were the densest 40-question stretch of each video.

## Setup

- New questions: 514 (how 246, why 245, weight 23) in 207 windows, built by [`scripts/build_hd_epic_q2_full.py`](../../scripts/build_hd_epic_q2_full.py). One how question was dropped for duplicate choices, the same filter as q40_v1.
- Run: the E1 config unchanged except name and data path. That means `per_field` (one question per call), temperature 0, 480p / 2 fps clips with the same `h264-crf30-veryfast-output-duration-v2` recipe, concurrency 1, one attempt per call. Runtime package revision `ad1ca97` on the Azure VM.
- Outcome: 514/514 calls succeeded, 0 invalid answers, cost $4.59.

## Files

- [`data/per_field.csv`](data/per_field.csv): 3,726 rows, made up of all 3,212 E1 rows (`experiment=E1`) plus the 514 new rows (`experiment=E1q2`). `window` is `q40_v1` or `new`.
- [`data/per_call.csv`](data/per_call.csv): the 514 new calls only, with tokens, cost and window. `call_latency_s` includes local clip preparation, so it is not comparable to E1's API time. E1 calls are in the [E1 results](../hd_epic_q40_v1_gemini35_flash_lite_20261020/data/per_call.csv).
- [`summary.json`](summary.json): every table above.
- To recompute from the repository root (Python 3.9+, no packages):

```bash
python results/hd_epic_q2_full_v1_gemini35_flash_lite_20261022/analyze.py
```

- [`collect.py`](collect.py) rebuilt the two CSVs from the VM run directory. The raw run (prompts, responses, wire inputs) stays on the VM at `~/Capstone-14/outputs/hd_epic_q2_full_20261022/`.
