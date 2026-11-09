# CC4D k = 16 / 32 Experiment Design (Draft)

## Purpose

Run Qwen3-VL-32B and Gemini 4 Argon with 16 and 32 questions per call. Two questions to answer:

1. Does asking 16 or 32 questions per call lower accuracy compared with 8? (effect of k)
2. If accuracy drops, is it because questions are farther from the end of the prompt, or because of the question count itself? (separate position from batch size, following A3 in HD-EPIC's `error_analysis.md`)

## Data limits

- 125 recordings have at least 16 usable localization steps. Only **44 recordings (7 recipes)** have at least 16 usable duration steps (excluding steps whose description contains a time expression and steps within ±2 s of a length cut point). These are exactly the 44 used by E5 duration.
- The feasible approach is to **ask the same step in two contents** (one duration question, one localization question): 16 steps × 2 = 32 questions.

## Design

**Reports:** the 44 E5 duration reports (14 of them are also in E5 localization).
**Answer types:** bool, single, seconds (as in E5). Within one call, the duration and localization questions use the same type.
**Target questions (the only ones scored):** the 16 duration facts E5 already has. They have single-content results at k = 1 / 4 / 8 / 16 for Qwen and Flash-Lite.
**Filler questions:** the localization questions on the same 16 steps (one per type).

| Condition | Call content | Calls per report per type | Purpose |
| --- | --- | --- | --- |
| Single k = 8 | 8 duration questions | 2 | Baseline (Qwen has it; Argon needs it) |
| Single k = 16 | 16 duration questions | 1 | Baseline (Qwen has it; Argon needs it) |
| Mixed k = 16 | 8 duration + 8 localization | 2 | Separates the effect of mixing itself |
| Mixed k = 32 | 16 duration + 16 localization | 1 | The condition under test |

Every condition is run forward and reversed. k = 1 and 4 are not rerun; k = 8 is the baseline. Qwen already drops about 4.6 pt from k = 1 to 8, so the conclusion is "from 8 upward" and is not compared with k = 1.

## What to compare

1. **Mixing effect:** the same duration fact, single k = 16 vs mixed k = 16. Q4 found mixing does not hurt; this checks whether that still holds at 16 questions.
2. **Effect of k:** the same duration fact, mixed k = 16 vs mixed k = 32 (same mix, only the question count differs).
3. **Position effect:** pool k = 8, 16, 32 and fit a conditional logistic regression `correct ~ batch size + log2(questions from the end + 1)` with question fixed effects; intervals bootstrapped over reports.
4. **Order:** forward vs reversed (Qwen got worse reversed in Q6, about −19 pt for localization at k = 16).

## Scale

| Model | Content | Calls (44 reports × 3 types × forward + reversed) |
| --- | --- | --- |
| Qwen3-VL-32B | Mixed k = 16 (2 calls) + mixed k = 32 (1 call) | 44 × 3 × 3 × 2 = 792 |
| Argon | The above + single k = 8 (2 calls) + single k = 16 (1 call) | 44 × 3 × 6 × 2 = 1,584 |

Qwen keeps its three video-length resolution classes (≤ 11 min, 11–17 min, > 17 min); every result is tagged with its length class and resolution.

## Expected cost and time

Estimated from earlier measured runs: Qwen about one working day of GPU time; Argon about $100–$300 and half a day. Both can run at the same time (different machines / quotas).

## Risks and limitations

1. **44 reports, 7 recipes.** Conclusions hold only for this set, and intervals will be wider than in the main experiment.
2. **Same step asked twice.** The filler questions ask about the same steps, so they may cue each other or interfere; this cannot be fully ruled out. It is a design trade-off and has not been validated.
3. **Mixing and k are partly confounded.** Mixed k = 16 is used as a midpoint to separate them; if mixed k = 16 already differs from single k = 16, read the k = 32 result conservatively.
4. **Argon ceiling.** Argon's localization is already 95–97% at k ≤ 8; the duration target questions have more headroom (duration single about 75%).
