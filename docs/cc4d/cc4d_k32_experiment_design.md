# CC4D Full-Report Batching Experiment Design (Draft)

## Question

For the same report, how much does accuracy differ between asking **all questions at once** and asking them **in batches**? Does question **position** (distance from the end of the prompt) matter? Which answer type and which content deteriorates fastest with position?

This follows the HD-EPIC q40 study (multiple choice only) and moves to typed fields of real reports (bool, multi-choice, seconds, integer). Models: Qwen3-VL-32B and Gemini 4 Argon.

## Method

- **Reports:** full reports from `cc4d_dense_v1`. Each step has three questions (done correctly?, what mistakes?, how many seconds?), plus 4–5 recording-level questions (which steps were skipped, order violated, steps done, total duration, longest step).
- **Batch size:** split each report's questions, in their original order, into batches of **8, 16 or 32**, plus **all at once**.
- **Order:** every batch size is run **forward** and **reversed** (the whole question order flipped).
- **Videos:** the 64 recordings of the main experiment (19 recipes, 12.5 min on average).

| Batch size | Calls per report (average) |
| --- | --- |
| 8 | about 7 |
| 16 | about 4 |
| 32 | about 2 |
| All | 1 |

A report has 52.7 questions on average (range 32–79), so "all" means k of about 32 to 79.

### Example: a video with 18 steps (4 recording-level questions, 4 + 18×3 = 58 questions)

Question order (forward):

```
No.:   1  2  3  4 | 5  6  7 | 8  9 10 | ... | 56 57 58
Kind:  R  R  R  R | c  e  d | c  e  d | ... |  c  e  d
        recording    step 1    step 2         step 18
(R = recording-level, c = done correctly?, e = what mistakes?, d = how many seconds?)
```

Forward split:

```
k = 8   (8 calls)
 [1-8] [9-16] [17-24] [25-32] [33-40] [41-48] [49-56] [57-58]
                                                      └ only 2 questions

k = 16  (4 calls)
 [1-----------16] [17----------32] [33----------48] [49-----58]
                                                    └ 10 questions

k = 32  (2 calls)
 [1--------------------------32] [33-----------------58]
                                  └ 26 questions

All     (1 call)
 [1-----------------------------------------------58]
```

Reversed: flip the whole question list first (58, 57, …, 1), then split the same way. Each call then holds different questions (for example the last k = 8 batch is questions 57–58 forward but questions 2 and 1 reversed), and the recording-level questions move from the start to the end.

Position of the same question under each condition (question 20, step 6 "done correctly?"; "from the end" = how many questions come after it):

| Condition | Call | Position in call | From the end |
| --- | --- | --- | --- |
| Forward k = 8 | 3rd | 4 / 8 | 4 |
| Forward k = 16 | 2nd | 4 / 16 | 12 |
| Forward k = 32 | 1st | 20 / 32 | 12 |
| Forward all | 1st | 20 / 58 | 38 |
| Reversed k = 8 | 5th | 7 / 8 | 1 |
| Reversed k = 16 | 3rd | 7 / 16 | 9 |
| Reversed k = 32 | 2nd | 7 / 26 | 19 |
| Reversed all | 1st | 39 / 58 | 19 |

The same question appears under 8 conditions with different combinations of batch size and distance from the end (for example forward k = 16 and k = 32 are both 12 from the end but have different batch sizes), so the two can be estimated separately.

## Question composition

Average per report over the 64 reports:

| Content | Answer type | Questions |
| --- | --- | --- |
| Done correctly? (correct) | bool | 16.6 |
| What mistakes? (error types) | multi-choice | 16.5 |
| How many seconds? (duration) | seconds | 14.9 |
| Recording-level (skipped, order, steps done, total time, longest step) | multi-choice, bool, integer, seconds, single-choice | 4.7 |

The three main types are about a third each and alternate within a report (for each step one correct, one error types, one duration question), so the type mix around any position is roughly even. Recording-level questions always come first (last when reversed).

## Analysis

1. **Whole report vs batches:** pair the same question across 8, 16, 32 and all; compare overall accuracy.
2. **Position effect:** conditional logistic regression `correct ~ batch size + log2(questions from the end + 1)` with question fixed effects, bootstrap over reports (as in A3 of HD-EPIC's `error_analysis.md`). Forward and reversed put the same question at different positions, which separates position from the question itself.
3. **By type and by content:** estimate the position slope separately for bool, multi-choice and seconds, and for correct, error, duration and recording-level, to see which deteriorates fastest.
4. **Order:** forward vs reversed.
5. **Account for answer bias:** earlier results showed error-content accuracy comes mainly from no-mistake steps (Q2), so report normalised accuracy and split mistake and no-mistake steps.

## Scale and cost (rough)

About **1,750 calls** per model (64 reports × four batch sizes × forward + reversed).

- **Argon:** about 57k input tokens per call, about 100M tokens in total. At the introductory price about **$50–$205** (depends on cache hits); about **4–6 hours** (concurrency 4).
- **Qwen3-VL-32B:** self-hosted, no API cost. Extrapolated from the measured E5 k = 16 latency (about 97 s, 210k input tokens): about **1–1.5 days**, **40–70 A100 hours**. These videos are shorter (12.5 min on average), so most do not need reduced resolution.
- With 125 videos instead (17.6 min on average): about 3,740 calls, Argon about $160–$630.

## Risks and limitations

1. **Cannot separate the effect of mixing itself.** Types and contents are mixed, so only the overall effect and position can be seen, not whether a difference is due to the mixing.
2. **Position is linked to content.** Questions follow step order; reversing helps separate them, but not completely.
3. **Few recording-level questions.** About 5 per report, always at the start or the end, so their position slope will be unstable.
