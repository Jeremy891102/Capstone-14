# Error analysis: when and how Gemini 3.5 Flash Lite gets questions wrong

Model Gemini 3.5 Flash Lite | Question pool hd_epic_q40_v1 | 2026-10-21

## Contents

- [Summary](#summary)
- [Background: data and terms](#background-data-and-terms)
- **A. When errors happen**
  - [A1 Segment duration](#a1-segment-duration)
  - [A2 Action verb](#a2-action-verb)
  - [A3 Batch size and position in the prompt](#a3-batch-size-and-position-in-the-prompt)
- **B. What the errors look like**
  - [B1 Errors are consistent, not random](#b1-errors-are-consistent-not-random)
  - [B2 How far off, and what gets picked](#b2-how-far-off-and-what-gets-picked)
  - [B3 Preference for the longest option](#b3-preference-for-the-longest-option)
  - [B4 Letter bias](#b4-letter-bias)
- [Other observations](#other-observations)
- [Suggested follow-up experiments](#suggested-follow-up-experiments)

---

## Summary

The analysis covers **localization questions**. Each question gives a video clip and asks when an action happened. The model picks one of five time-segment options, so chance accuracy is 20%.

| | Question | Finding | Cannot claim |
|---|---|---|---|
| **A1** | Are shorter segments harder? | Yes. 18% accuracy below 0.5 s, 62.5% above 8 s | That 2 FPS sampling is the cause |
| **A2** | Does the action verb matter? | Yes. At equal duration, pick stays near chance; rinse is easiest | That the action itself is the cause |
| **A3** | Why does asking more questions per call hurt? | Larger batches push most questions far from the end of the prompt. The drop works through position: each doubling of distance from the end multiplies the odds of a correct answer by 0.81. At equal distance from the end, batch size has no extra effect | Why questions near the end do better |
| **B1** | Are errors random? | No. The same question is often wrong every time, and often with the same wrong option | That the model holds a fixed wrong belief |
| **B2** | Near misses or far off? | Far off. The median wrong pick is 33.5 s from the correct segment | That the model picks distractors *because* they are real actions |
| **B3** | Is there an option-length bias? | Yes. Among errors, it favors the longest wrong option (31.1% vs 25% chance) | That length itself is the cause |
| **B4** | Is there a letter bias? | Yes. One question per call favors B (27.1% vs 19.6%); batched calls under-pick E | That the letter itself is the cause |

All findings are **correlational, not causal**. Each condition was run once, with one model.

---

# A. When errors happen

## A1 Segment duration

**Question**: Is the model more often wrong when the correct segment is short?

**Answer**: Yes, and the association is robust.

| Correct segment duration | Accuracy | Questions |
|---|---|---|
| < 0.5 s | 18.0% | 384 |
| 0.5–1 s | 25.0% | 857 |
| 1–2 s | 34.0% | 897 |
| 2–4 s | 40.5% | 454 |
| 4–8 s | 55.8% | 181 |
| > 8 s | 62.5% | 88 |

- 1,241 questions (43%) have a correct segment shorter than 1 s. This may be the main reason overall localization accuracy is only about 32%.
- The effect is not just B3's longest-option bias. Among questions where the correct answer is also the longest option, longer segments are still more accurate (28.6% below 1 s, 41.3% at 1–2 s, 52.1% above 2 s).

**Cannot answer**: Whether 2 FPS sampling is the cause. At 2 FPS there is one frame every 0.5 s, so an action shorter than 0.5 s may appear in one frame or none. This is a plausible guess the data does not test.

**Method**

- **Spearman correlation** (monotonic relation between duration and correctness): ρ = 0.23, p ≈ 5e-37.
- **Logistic regression** on log2 duration, with a bootstrap that resamples whole videos: each doubling of duration multiplies the odds of a correct answer by 1.53 (95% interval 1.41–1.68).

![Accuracy by segment duration; x-axis labels are bin upper bounds, n is question count](figures/error_a1_duration_accuracy.png)

Code and full output: section A1 of [error_trend_analysis.ipynb](error_trend_analysis.ipynb).

---

## A2 Action verb

**Question**: Do some actions (verbs) differ in difficulty, or do some just have shorter segments?

**Answer**: The verb matters. After controlling for duration, adding the verb still improves the model significantly (p ≈ 7e-16).

Verbs with at least 60 questions (7 verbs, 1,732 of 2,861 questions):

| Verb | Questions | Accuracy | Odds vs pick at equal duration |
|---|---|---|---|
| pick | 669 | 19% | 1 (reference) |
| move | 66 | 21% | 1.0 (not different from pick) |
| transfer | 89 | 24% | 1.3 (not different from pick) |
| put | 320 | 31% | 1.8 |
| shake | 77 | 34% | 1.9 |
| place | 417 | 39% | 2.6 |
| rinse | 94 | 63% | 5.6 |

- pick, move and transfer stay near chance even when segments are longer.
- rinse and place improve as segments get longer.

**Cannot answer**: Whether the remaining difference comes from the action itself. A verb carries many factors that were not tested: speed and displacement (pick is a fast hand movement that 2 FPS may catch only before and after), repeated occurrences of the same action that are hard to tell apart, similar distractors, and different scenes.

**Method**

- **Stratified heatmap**: verb × duration (< 1 s, 1–2 s, > 2 s), showing accuracy and n per cell. Do not over-read small cells.
- **Logistic regression** `correct ~ log2(duration) + verb`: including duration compares verbs at equal duration. A likelihood-ratio test checks whether adding the verb helps.

![Accuracy by verb and segment duration](figures/error_a2_verb_heatmap.png)

Code and full output: section A2 of [error_trend_analysis.ipynb](error_trend_analysis.ipynb).

---

## A3 Batch size and position in the prompt

**Question**: Accuracy drops when many questions share a call. Is that an effect of batch size itself, or because larger batches put most questions far from the end of the prompt?

**Answer**: Batch size is the root cause, but it works mainly through position: larger batch → longer prompt → most questions far from the end → lower accuracy. At equal distance from the end, batch size has no extra negative effect.

The same 2,336 questions were asked in E1–E5, with identical text, video, options and 32-question order. E2–E4 split the same 32-question list, in order, into calls of 4, 8 and 16:

| Questions per call | 1 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|
| Accuracy | 33.0% | 34.9% | 35.0% | 30.7% | 26.7% |

The order is fixed, but a question's distance from the end changes across conditions. Take question 20 of the 32-question list: in E5 and E4 it is 12 questions from the end, in E3 it is 4 from the end, and in E2 it is the last question. Comparing each question with itself separates position from batch size:

| Variable (odds ratio vs E1) | OR | 95% interval |
|---|---|---|
| **Each doubling of distance from the end** | **0.81** | **0.75–0.87** |
| Batch of 4 | 1.45 | 1.23–1.72 |
| Batch of 8 | 1.71 | 1.39–2.17 |
| Batch of 16 | 1.52 | 1.16–2.03 |
| Batch of 32 | 1.35 | 0.95–1.93 |

- **Closer to the end of the prompt is more accurate**: each doubling of distance from the end multiplies the odds by 0.81.
- **Batch size has no extra direct effect**: after controlling for position, no batch-size OR is below 1. As the last question of a prompt, a question does better inside a batch than asked alone.
- **The position effect does not depend on batch size**: the slope is nearly the same for 32 and for other batch sizes (interaction OR 0.98, p = 0.75).
- **So batch size lowers accuracy through position**: with 8 per call, every question is at most 7 from the end. With 32 per call, 24 questions are 8 or more from the end, and those are the ones that get worse. Batch size sets how far from the end a question can be.
- **In practice**: to avoid the drop, cap how far any question sits from the end; batch size is the lever for that. The current data supports at most about 8 questions per call.

Change in accuracy vs E1 (percentage points), grouped by distance from the end. At equal distance, batch sizes show similar changes:

| Questions from the end | 4 per call | 8 per call | 16 per call | 32 per call |
|---|---|---|---|---|
| 0 (last question) | +4.8 | +4.8 | +7.5 | +6.8 |
| 1 | +2.9 | +6.2 | +3.4 | +5.5 |
| 2–3 | 0.0 | +2.2 | +2.7 | −2.1 |
| 4–7 | | +0.2 | −1.4 | −4.1 |
| 8–15 | | | −5.9 | −7.0 |
| 16–31 | | | | −8.6 |

**Cannot answer**

- **Why questions near the end do better.** In this data, "near the end" is the same as "near the output-format instructions" and "near where the model starts writing its answer". They cannot be separated.
- **Whether small-batch questions far from the end would also get worse.** Distances of 8 or more occur only in 16- and 32-question prompts, so "small batches behave the same" is a model extrapolation, not an observation.
- **Why the last few questions in a batch beat asking alone.**
- **Causality.** The order was never shuffled, and a question's neighbors differ across conditions. Interference between questions was not tested separately.

**Method**

- **Conditional logistic regression** `correct ~ batch size + log2(questions from end + 1)` with **question fixed effects**. Each question has its own baseline difficulty, so coefficients come only from changes in the same question's correctness across conditions. The 985 questions that are right all five times or wrong all five times carry no information and drop out.
- **Position form chosen by AIC** (lower fits better): log2(questions from end + 1) 5108.6; not in last 8 (0/1) 5115.5; questions from end, linear 5117.2; relative position 5121.4; no position 5137.8. Relative position fits worst, because what protects a question is being in roughly the last 8, an absolute count, not being in the second half of the prompt. In linear form, "count from start" and "count from end" fit identically (batch size is in the model, so they differ by a constant). A nonlinear form is needed to show the effect is about the end.
- **Intervals**: 200 bootstrap resamples of whole reports (73), refitting each time.
- **Interaction**: add position × batch-of-32 and test the slope difference with a likelihood-ratio test.

![Change in accuracy vs E1 by distance from the end of the prompt, one line per batch size](figures/error_a3_position.png)

Code and full output: section A3 of [error_trend_analysis.ipynb](error_trend_analysis.ipynb). The numbers here come from A3-2, the main analysis; A3-1 is a preliminary split of the 32-question list into four blocks.

---

# B. What the errors look like

## B1 Errors are consistent, not random

**Question**: Are errors luck of the draw, or does the same question fail repeatedly in the same way?

**Answer**: Not luck. Two "random" hypotheses are rejected: (1) correctness is independent across runs; (2) wrong answers are picked uniformly among the wrong options.

**Times each question was answered correctly** (same 2,336 questions, answered once in each of E1–E5):

| Times correct | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| Observed | **821** | 504 | 344 | 278 | 225 | **164** |
| If independent | 336 | 798 | 756 | 355 | 83 | 8 |

Always-wrong questions are 2.4 times the independent expectation, and always-right questions 20 times. Difficulty is mostly set by the question.

**For the 821 always-wrong questions, how many different wrong letters were picked across the five runs**:

| Distinct letters | 1 (same every time) | 2 | 3 | 4 |
|---|---|---|---|---|
| Observed | **162** | **419** | 211 | 29 |
| If guessing among 4 wrong options | 3 | 144 | 481 | 192 |

162 questions got the same wrong option all five times; random guessing predicts about 3. For 85% of these questions, one wrong letter appears 3 or more times.

**Cannot answer**

- Whether the model holds a fixed wrong belief. The five conditions share the same video and options, so they are not independent experiments. The first test only shows that questions differ in difficulty.
- The second test's baseline is uniform guessing, but B3 and B4 show shared preferences for long options and certain letters. Those preferences alone would make runs pick the same option, so "drawn to a specific wrong option" cannot be attributed entirely to question content.

**Method**

- **Permutation test**: shuffle each condition's correctness across questions 2,000 times (keeping each condition's accuracy) and count always-wrong questions. Observed 821; shuffled mean 336; no shuffle reached the observed value (p ≤ 0.0005).
- **Binomial test**: under uniform guessing among 4 wrong options, about 3 questions would pick the same one five times; observed 162 (p ≈ 2e-216).

![Times correct across five conditions: observed vs independent](figures/error_b1_correct_counts.png)

![Distinct wrong letters for always-wrong questions: observed vs random guessing](figures/error_b1_wrong_letters.png)

Code and full output: section B1 of [error_trend_analysis.ipynb](error_trend_analysis.ipynb).

---

## B2 How far off, and what gets picked

**Question**: When wrong, is the model slightly off in time, or picking a different action entirely?

**Answer**: Far off; it mostly picks a different action.

- The median midpoint distance between the chosen and correct segments is 33.5 s, about 7% of the clip length. Only 19% are within 5 s; 36% are more than 60 s away.
- For comparison, the four wrong options have a median distance of 38.4 s (8.0%). The model's picks are only slightly closer than random.
- There is a small pull toward nearby options: the closest wrong option is picked 30.1% of the time vs 25% by chance (p ≈ 4e-7), only 5 points more.
- 35% of wrong options are exactly the correct segment of another question in the same report, i.e. a real annotated action in the video.

**Cannot answer**

- Whether the model picks an option *because* it is a real action: real and non-real distractors were not compared.
- What the other 65% of wrong options are. They may be real annotated actions not sampled as questions, or other spans; the data cannot tell.
- Relative distance is divided by **the clip length the model saw** (205–886 s, median about 500 s), not the full source video, whose length was not recorded.

**Method**

- **Descriptive statistics**: distance in seconds and relative distance (distance ÷ clip length). Clip lengths vary, so both are reported.
- **Binomial test**: with 4 wrong options, random picking chooses the closest one 25% of the time.

![Relative distance to the correct segment: picked wrong option vs all wrong options](figures/error_b2_wrong_distance.png)

Code and full output: section B2 of [error_trend_analysis.ipynb](error_trend_analysis.ipynb).

---

## B3 Preference for the longest option

**Question**: Does the model prefer options with longer time spans?

**Answer**: Yes. The preference is clear and rises monotonically with length rank.

| | Rate | Chance | p |
|---|---|---|---|
| All questions: picked the longest option | 27.5% (787 / 2,861) | 20% | 5e-22 |
| Wrong answers only: picked the longest wrong option | 31.1% (602 / 1,933) | 25% | 1e-9 |

Among wrong answers, ordered from shortest to longest wrong option, pick rates are 18.5%, 24.0%, 26.4% and 31.1%.

**Cannot answer**: Whether option length itself is the cause. Length may stand in for another factor, such as distance from the correct time, more frames in longer spans (2 FPS), or how distractors were generated. The tests only reject uniform random picking.

**Method**

- Option lengths exist only in the prompt text, so they are parsed for all five options from `data/frozen_requests/E1.jsonl.gz`.
- **Two binomial tests**: (1) picked the longest option, chance 20%; (2) among wrong answers, picked the longest wrong option, chance 25%.

![Among wrong answers, pick rate by length rank of the wrong option](figures/error_b3_longest_option.png)

Code and full output: section B3 of [error_trend_analysis.ipynb](error_trend_analysis.ipynb).

---

## B4 Letter bias

**Question**: Does the model favor certain letters?

**Answer**: Yes, and the bias changes with how questions are asked.

Ground truth is spread evenly across letters, but the model's picks are not:

| | A | B | C | D | E |
|---|---|---|---|---|---|
| Ground truth (E1) | 20.2% | 19.6% | 20.2% | 20.3% | 19.6% |
| Model, E1 (1 per call) | **14.8%** | **27.1%** | 19.0% | 19.8% | 19.3% |
| E2 (4 per call) | 21.5% | 22.9% | 18.1% | 19.6% | 17.9% |
| E3 (8 per call) | 23.1% | 22.6% | 17.7% | 20.0% | **16.6%** |
| E4 (16 per call) | 19.9% | 20.4% | 21.2% | 22.5% | **16.0%** |
| E5 (32 per call) | 18.6% | 20.3% | 22.3% | 22.0% | **16.7%** |

- With one question per call (E1), the model clearly favors B and avoids A: it picks B 27.1% of the time, while B is correct only 19.6% of the time.
- In batched calls the B preference disappears; instead, every condition under-picks E.
- The B preference is unrelated to option length: when the model picks B, B is not longer.

**Cannot answer**: Whether the letter itself is the cause. Option order was never shuffled, so letter position and option content are tied together.

**Method**

- **Chi-square goodness-of-fit test**: does the model's letter distribution match ground truth? χ² is 126 for E1 and 22–37 for the other conditions; all p < 0.001. Larger χ² means larger deviation.
- **Binomial test**: E1 picks B 776 / 2,861 times (27.1%) against the ground-truth B share of 19.6%, p ≈ 4e-22.

![Picked share minus ground-truth share per letter; 0 means no bias](figures/error_b4_letter_bias.png)

Code and full output: section B4 of [error_trend_analysis.ipynb](error_trend_analysis.ipynb).

---

## Other observations

- **Large differences between videos**: per-video localization accuracy ranges from 5% to 57% (median 32%). By participant it ranges from 26% to 39%.
- **Small effect of position in time**: accuracy is slightly higher when the correct answer is in the last 20% of the clip (36% vs about 31% elsewhere), and when the correct option is the latest one in time (39.9%).
- **No clear bias in How/Why questions**: neither shows a letter or length bias. Why questions are more accurate when the correct answer is the longest option (80.6% vs 67.4%), but only 31 such questions exist.
- **Too few Weight questions**: 11 of 16 are wrong, and options often differ by a gram or so (e.g. 229 g vs 230 g), which video alone can hardly resolve. Not enough to conclude anything.

## Suggested follow-up experiments

| To confirm | Approach |
|---|---|
| A1: whether short-segment errors come from the sampling rate | Rerun questions shorter than 1 s at a higher FPS |
| A2: whether pick is hard because it is fast with large displacement | Quantify motion within the segment with optical flow or hand tracking and add it to the regression; or rerun pick questions at a higher FPS |
| A3: why questions near the end do better | Rerun the 32-question prompt in reverse order to see whether the effect follows position or the question; move the output-format instructions before the questions to separate "near the format instructions" from "near where the answer starts" |
| B2: whether distractors are picked because they are real actions | Compare pick rates of real vs non-real distractors; check the original HD-EPIC annotations for the source of the other 65% |
| B3, B4: length and letter bias | Shuffle option order, or decouple option length from the actual segment, and rerun the same questions |
