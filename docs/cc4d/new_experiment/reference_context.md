# Reference questions × video control: does adding 7 reference questions only help when the video is present?

Models: Gemini 4 Argon, Qwen3-VL-32B. 
Scripts: `results/cc4d_reference.py` (output `cc4d_reference_out.txt`). Data build: `scripts/build_cc4d_reference.py`. Configs: `experiments/cc4d_ref/`. Per-question scores: `results/cc4d_reference_v1/`.

## Setup

- Targets: 64 recordings, 8 questions per recording for each of bool / multi-choice / seconds, **1,528 targets** in total. Each call answers one target question and only that question is scored.
- Reference questions: the other 7 questions of the target's k = 8 batch, shown as background only and not answered.
- Four conditions: {target alone, target + 7 references} × {with video, without video}. 8 runs (2 models) × 1,528 calls, all succeeded.
- **D = (context gain with video) − (context gain without video)**, paired by target, bootstrapped over recordings (10,000 draws). A positive D means the reference questions help more when the video is present.
- Every question allows the answer `not_visible` (same as the k32 experiment).

## Conclusions

- **Argon: adding reference questions does not help, and there is no evidence that it helps more with video.** Overall D = +1.0 [−2.3, 4.3], interval crosses 0. Accuracy drops slightly both with and without video (−2.3, −3.3). This is the opposite direction from the teammate's HD-EPIC result (about +8 pt in both conditions).
- **Qwen (with `not_visible` allowed): cannot answer the question.** Without video it answers `not_visible` almost every time, so accuracy reflects the abstention rate, not the information in the reference questions.
- **Qwen (rerun without `not_visible`): D = +1.2 [−1.0, 3.6], interval crosses 0.** No evidence that reference questions help more with video. Qwen is also less accurate with video (31.8%) than without (36.4%); see the section "Qwen: rerun without `not_visible`".

## Argon

Accuracy in %; the change from adding references is in parentheses.

| Type | n | With video: alone → +refs | Without video: alone → +refs | D [95% CI] |
| --- | --- | --- | --- | --- |
| All | 1,528 | 61.3 → 59.0 (−2.3) | 34.4 → 31.1 (−3.3) | +1.0 [−2.3, 4.3] |
| Bool | 510 | 69.2 → 64.1 | 56.3 → 51.0 | +0.2 [−6.4, 6.7] |
| Multi-choice (which mistakes) | 510 | 50.2 → 48.4 | 22.5 → 26.7 | −5.9 [−11.4, −0.2] |
| Seconds | 508 | 64.4 → 64.4 | 24.2 → 15.6 | +8.7 [3.5, 14.0] |

- The positive D for seconds comes from **seconds accuracy dropping 8.7 pt without video when references are added** (`not_visible` answers 54 → 77); with video it does not change. It is not "references help more with video".
- The negative D for multi-choice has an interval close to 0. Three question types are tested separately with no multiple-comparison correction, so treat these as hints only.
- Excluding abstentions (keeping only the 1,390 targets that no condition answered `not_visible`) does not change the conclusion: overall D = +1.2 [−2.4, 4.7], seconds +9.8 [3.5, 16.3]. So the Argon result is not caused by abstention.

## Qwen3-VL-32B (with `not_visible` allowed)

| Type | With video: alone → +refs | Without video: alone → +refs | D [95% CI] |
| --- | --- | --- | --- |
| All | 30.2 → 32.8 | 14.2 → 20.5 | −3.8 [−5.8, −1.8] |
| Bool | 45.9 → 51.4 | **0.4** → 19.6 | −13.7 [−18.6, −8.9] |
| Multi-choice | 29.0 → 31.0 | 42.2 → 40.4 | +3.7 [0.4, 7.1] |
| Seconds | 15.7 → 15.9 | **0.0** → 1.6 | −1.4 [−4.2, 1.4] |

- **Without video and with the target alone, Qwen answers `not_visible` on 503/510 bool and 508/508 seconds questions**, which is why accuracy is 0.4% and 0.0%. With references added, `not_visible` on bool drops to 316/510 and accuracy jumps to 19.6%. That +19 pt is just "abstaining less", so D = −13.7 cannot be read as an effect of the video or the references.
- After excluding abstentions, only 517 / 1,528 targets were never answered `not_visible`; 510 of them are multi-choice, bool has 7 and seconds has 0. On the remaining multi-choice questions D = +3.7 [0.4, 7.1], but the no-video accuracy of 42.2% is already close to the 44.9% ceiling of always giving the most common answer ("no mistake"), so this looks more like answer bias.

## Limitations

1. **Target position and format differ from the teammate's setup:** the target is always placed after the references (they keep its original position), and the references have only the question and options, without the answer-format hint. D measures the whole protocol, not the semantic content of the questions alone.
2. **`not_visible` allowed:** this changes accuracy a lot without video, especially for Qwen. Qwen was therefore rerun with `not_visible` disabled (see the section "Qwen: rerun without `not_visible`"); Argon was not rerun (its abstention rate is low and the conclusion is unchanged after excluding abstentions).
3. **Three separate tests, one per question type,** with no multiple-comparison correction; treat intervals close to 0 as hints.
4. **An interval that crosses 0 does not mean there is no difference.**

## Qwen: rerun without `not_visible`

Same 1,528 targets and reference groups. No question allows `not_visible` and the answer-format hint no longer mentions it (`--no-nv`; design in the last section of the design document). All 4 Qwen runs of 1,528 calls succeeded.

| Type | With video: alone → +refs | Without video: alone → +refs | D [95% CI] |
| --- | --- | --- | --- |
| All | 31.8 → 33.2 (+1.4) | 36.4 → 36.6 (+0.2) | +1.2 [−1.0, 3.6] |
| Bool | 46.1 → 51.0 | 47.1 → 48.0 | +3.9 [−0.8, 8.6] |
| Multi-choice | 33.1 → 32.9 | 44.1 → 42.0 | +2.0 [−1.6, 5.5] |
| Seconds | 16.1 → 15.7 | 17.9 → 19.7 | −2.2 [−5.6, 1.2] |

- **Every D interval crosses 0.** The first-round D = −13.7 on bool disappears, which confirms it was caused by the abstention rate.
- **Without video, Qwen relies on fixed answers:** with the target alone, 423/510 bool questions are answered "done correctly" (83%; the true rate is 54%), 474/510 multi-choice questions are answered "no mistake" (93%), and 168/508 seconds questions are answered 15 s. Accuracy is therefore 47.1% (bool) and 44.1% (multi-choice), right at the fixed-answer ceilings (54.3%, 44.9%).
- **Qwen with video is worse than without:** with the target alone, 31.8% < 36.4% (multi-choice 33.1 < 44.1). With video it says "no mistake" less often (162/510) and names specific mistakes instead (for example C on 126 questions), but is mostly wrong, so it loses to the always-"no mistake" strategy. This agrees with the no-video experiment: Qwen gains little from the video.
- Adding references also changes the share of fixed answers: without video, bool "done correctly" 423 → 302 and multi-choice "no mistake" 474 → 438. Accuracy changes come mainly from this, not from new information.
- Accuracies of the two rounds cannot be subtracted from each other (the difference is itself the effect of allowing abstention).
