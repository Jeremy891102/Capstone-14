# CC4D Full-Report Batching Experiment Design (Draft)

## Question

For the same report, how much does accuracy differ between asking **all questions at once** and asking them **in batches**? Does question **position** (distance from the end of the prompt) matter? Which answer type and which content deteriorates fastest with position? For steps that were not done, does the model **hallucinate** (fill in an answer instead of abstaining), and does that depend on position and batch size?

This follows the HD-EPIC q40 study (multiple choice only) and moves to typed fields of real reports (bool, multi-choice, seconds, integer). Models: Qwen3-VL-32B and Gemini 4 Argon.

## Method

- **Reports:** full reports from `cc4d_dense_v1`. Each step has three questions (done correctly?, what mistakes?, how many seconds?), plus recording-level questions (order violated, steps done, total duration, longest step). **`skipped_steps` (which steps were not done) is removed:** its options contain all the missing steps and would leak the answers to the abstention questions.
- **Abstention questions:** for each step that was not done (annotated Missing Step, no footage), add the same three questions as for a normal step. These are real skipped steps in the data, not invented ones. The correct answer is `not_visible`; any other answer counts as a hallucination. Each is placed at its step's position in the recipe, mixed in with the other questions.
- **Abstention setting:** every question (normal steps and recording-level too) allows the answer `not_visible`, and the prompt treats all questions alike. The data contract requires this: setting it only on the abstention questions would give the answer away. The option lists of choice questions do not change.
- **Batch size:** split each report's questions, in their original order, into batches of **8, 16 or 32**, plus **all at once**.
- **Order:** every batch size is run **forward** and **reversed** (the whole question order flipped).
- **Videos:** the 64 recordings of the main experiment (19 recipes, 12.5 min on average); 47 of them have steps that were not done, 108 in total.

A report has 56.8 questions on average (range 31–83), so "all" means k of about 31 to 83.

| Batch size | Calls per report (average) |
| --- | --- |
| 8 | about 7.5 |
| 16 | about 4 |
| 32 | about 2.2 |
| All | 1 |

### Example: a video with 18 steps (3 recording-level questions, 3 + 18×3 = 57 questions)

Assume all 18 steps were done. Question order (forward):

```
No.:   1  2  3 | 4  5  6 | 7  8  9 | ... | 55 56 57
Kind:  R  R  R | c  e  d | c  e  d | ... |  c  e  d
        recording  step 1    step 2         step 18
(R = recording-level, c = done correctly?, e = what mistakes?, d = how many seconds?)
```

Forward split:

```
k = 8   (8 calls)
 [1-8] [9-16] [17-24] [25-32] [33-40] [41-48] [49-56] [57]
                                                      └ only 1 question

k = 16  (4 calls)
 [1-----------16] [17----------32] [33----------48] [49-----57]
                                                    └ 9 questions

k = 32  (2 calls)
 [1--------------------------32] [33-----------------57]
                                  └ 25 questions

All     (1 call)
 [1-----------------------------------------------57]
```

Reversed: flip the whole question list first (57, 56, …, 1), then split the same way. Each call then holds different questions (for example the last k = 8 batch is question 57 forward but question 1 reversed), and the recording-level questions move from the start to the end. If the video has steps that were not done, their three questions are inserted at the step's position and split in the same way.

Position of the same question under each condition (question 19, step 6 "done correctly?"; "from the end" = how many questions come after it):

| Condition | Call | Position in call | From the end |
| --- | --- | --- | --- |
| Forward k = 8 | 3rd | 3 / 8 | 5 |
| Forward k = 16 | 2nd | 3 / 16 | 13 |
| Forward k = 32 | 1st | 19 / 32 | 13 |
| Forward all | 1st | 19 / 57 | 38 |
| Reversed k = 8 | 5th | 7 / 8 | 1 |
| Reversed k = 16 | 3rd | 7 / 16 | 9 |
| Reversed k = 32 | 2nd | 7 / 25 | 18 |
| Reversed all | 1st | 39 / 57 | 18 |

The same question appears under 8 conditions with different combinations of batch size and distance from the end (for example forward k = 16 and k = 32 are both 13 from the end but have different batch sizes), so the two can be estimated separately.

## Question composition

Average per report over the 64 reports:

| Content | Answer type | Questions |
| --- | --- | --- |
| Done correctly? (correct) | bool | 16.6 |
| What mistakes? (error types) | multi-choice | 16.5 |
| How many seconds? (duration) | seconds | 14.9 |
| Recording-level (order, steps done, total time, longest step) | bool, integer, seconds, single-choice | 3.7 |
| Abstention questions for steps not done (1.7 each of the three) | bool, multi-choice, seconds | 5.1 |

The three main types are about a third each and alternate within a report (for each step one correct, one error types, one duration question), so the type mix around any position is roughly even. Recording-level questions always come first (last when reversed).

## Analysis

1. **Whole report vs batches:** pair the same question across 8, 16, 32 and all; compare overall accuracy.
2. **Position effect:** conditional logistic regression `correct ~ batch size + log2(questions from the end + 1)` with question fixed effects, bootstrap over reports (as in A3 of HD-EPIC's `error_analysis.md`). Forward and reversed put the same question at different positions, which separates position from the question itself.
3. **By type and by content:** estimate the position slope separately for bool, multi-choice and seconds, and for correct, error, duration and recording-level, to see which deteriorates fastest.
4. **Order:** forward vs reversed.
5. **Hallucination (abstention):** for steps not done, the share of answers other than `not_visible`, and how it changes with batch size, distance from the end and answer type. Also the reverse, the share of done steps answered `not_visible` (false abstention), because a model can lower its hallucination rate just by saying "not visible" more often (as Flash-Lite did in Q5).
6. **Account for answer bias:** earlier results showed error-content accuracy comes mainly from no-mistake steps (Q2), so report normalised accuracy and split mistake and no-mistake steps.

## Scale and cost (rough)

About **1,900 calls** per model (64 reports × four batch sizes × forward + reversed).

- **Argon:** about 58k input tokens per call, about 110M tokens in total. At the introductory price about **$55–$220** (depends on cache hits); about **4–6 hours** (concurrency 4).
- **Qwen3-VL-32B:** self-hosted, no API cost. Extrapolated from the measured E5 k = 16 latency (about 97 s, 210k input tokens): about **1–1.5 days**, **40–70 A100 hours**. These videos are shorter (12.5 min on average), so most do not need reduced resolution.

## Risks and limitations

1. **Cannot separate the effect of mixing itself.** Types and contents are mixed, so only the overall effect and position can be seen, not whether a difference is due to the mixing.
2. **Position is linked to content.** Questions follow step order; reversing helps separate them, but not completely.
3. **Few recording-level questions.** About 4 per report, always at the start or the end, so their position slope will be unstable.
4. **Small hallucination sample.** Only 47 videos and 108 steps not done (324 questions); treat as a secondary result.
5. **Steps not done are not random decoys.** They are steps that were really skipped, so the abstention rate also reflects whether the model can tell the person skipped the step, not only hallucination.
6. **Every question allows abstention.** The model's abstention rate can affect accuracy on the other questions, so results cannot be compared directly with earlier runs without abstention, only across the conditions of this experiment.
7. **Last batch is shorter.** With batches of 8, 16, 32 the last batch has fewer questions (in the example the last k = 8 batch has 1), so k is not strictly fixed.
8. **Output length.** "All" needs up to 83 answers in one reply; Qwen's current output cap is 2,048 tokens, which must be checked first.
9. **Qwen context.** The video token budget is unchanged and the extra question text is small, so it should fit, but run a gate test on the longest video.
10. **Argon introductory price ends 2026-11-30.** Best to finish before then.
11. **No multiple-comparison correction.** Treat p between 0.01 and 0.05 as a lead only.

## Next steps

1. Build the data: remove `skipped_steps`, add the abstention questions for steps not done, set `allow_not_visible` on every question, and confirm the scoring rules (`docs/scoring_rules.md`) work for all fields.
2. Run a small test on 5 reports with each model to check output length, cost and latency.
3. Videos are fixed to the 64 main-experiment recordings.
