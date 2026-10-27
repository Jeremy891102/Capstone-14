# CC4D answer-type experiment: question rules (draft, 2026-10-26)

Chinese version: `docs/cc4d_format_questions.zh.md`. Both say the same thing.

How each answer type's questions are generated from the CaptainCook4D raw annotations. The
experiment design is in `docs/cc4d_answer_type_proposal.md`, the scoring rules in
`docs/scoring_rules.md`.

## Principles

- **Facts first; the answer type is only how a fact is asked.** The same fact is asked once in
  each answer type, in separate calls. Within one call, each step appears only once.
- **Answer types are assigned without looking at the answer, and option lists do not change with
  the answer.**
- Questions are in English. Step names follow `cc4d_dense_v1` (description text, ordinal for
  repeated steps, no `step_id`).

## Model input

- **Video:** the whole recording, from 0 s to 5 s after the last annotated step, fps = 2.
- **Context:** `task_name`, `recipe`, `error_definitions`.

## Annotations used

| Field | Use |
| --- | --- |
| `start_time` / `end_time` | When and how long a step was done; `-1` means no footage |
| `errors[].tag` | Error category; `Missing Step` means not done |
| `errors[].description` | Reference answer for open_text |
| `modified_description` | Shown to the judge only |
| task graph `edges` | Finding questions the recipe leaks. Node ids differ from `step_id`; match by description text |

## Matrix

| Content \ type | bool | single | multi | seconds | int (count) | open_text |
| --- | --- | --- | --- | --- | --- | --- |
| Error (4,696 steps) | ✓ | 6 options | 6 options | — | ✓ | ✓ |
| Duration (3,763 steps) | ✓ | 4 ranges | 8 steps | ✓ | 8 steps | — |
| Localization (5,049 steps) | ✓ | 5 ranges | 8 steps | ✓ | 8 steps | — |

- Step counts are the pool over all 350 reports of `cc4d_dense_v1`. The reports used (64 for the
  main experiment, 125 / 44 for E5) are in the proposal.
- Empty cells: the annotations give no time for when a mistake happened; open questions on
  duration and localization are hard to score.
- int is always a count, never seconds.
- "8 steps" means one question lists 8 steps, so each report gives one such question per row.

## Error

Only 5 categories: Preparation, Measurement, Technique, Timing, Temperature. Order needs the
order of the whole video, so it moves to the supplementary content. Missing Step becomes a
hallucination item.

| Steps (5,295 in `cc4d_dense_v1`) | Count | Handling |
| --- | --- | --- |
| Filmed, no mistake | 3,591 | asked |
| Filmed, one of the 5 categories | 1,118 | asked; Order ignored |
| Only Order mistakes | 334 | excluded (cannot count as no mistake) |
| Other | 6 | excluded |
| Reference too vague (list in `scoring_rules.md`) | 13 | excluded |
| Missing Step, no footage | 237 | hallucination item, target `null` |
| Missing Step but has footage | 3 | excluded |
| No footage, no Missing Step tag | 6 | excluded |

4,696 steps in all. Tags per step: 0 on 3,591, 1 on 962, 2 on 137, 3 on 6. Answering "no
mistake" everywhere scores 76.5%, so the experiment samples 4 with and 4 without a mistake and
reports the two groups separately.

| Type | Question | Target |
| --- | --- | --- |
| bool | Was the step "X" done correctly? | No tag from the 5 categories |
| single | Which of these mistakes was made in the step "X"? If several were made, pick any one. 5 categories + No mistake | Any true mistake is correct |
| multi | Which of these mistakes were made in the step "X"? Same 6 options | Set of tags; No mistake may only be chosen alone |
| int | How many different kinds of mistakes were made in the step "X"? | 0–3 |
| open_text | Compared with the recipe, what did the person do differently in the step "X"? Answer "none" if the step was done as written. | Mistake descriptions, or none |

## Duration

Pool: the 5,049 filmed steps, minus steps whose description states a time (the text alone gives
a 40% chance of being right): 4,527. Minus steps within ±2 s of the cut points 17.1 / 31.3 /
56.5 s: 3,763.

| Type | Question | Target |
| --- | --- | --- |
| bool | Did the step "X" take longer than 31 seconds? | > 31.3 s, about half yes |
| single | How long did the step "X" take? 4 ranges | The range it falls in; ranges hold similar counts |
| multi | Which of these steps took longer than 31 seconds? 8 steps | Steps > 31.3 s |
| seconds | How many seconds did the step "X" take? | `end − start` |
| int | Among these steps, how many took longer than 31 seconds? Same 8 steps | Number of steps |

## Localization

Pool: the 5,049 filmed steps.

| Type | Question | Target |
| --- | --- | --- |
| bool | Was the step "X" being done between t1 and t2? | Whether they overlap; half overlap |
| single | When was the step "X" done? 5 ranges | The true range; the other 4 come from other steps and do not overlap it |
| multi | Which of these steps were being done between t1 and t2? 8 steps | Steps that overlap |
| seconds | At what time did the step "X" start? | `start_time`, tolerance `max(2, 0.25 × step length)` |
| int | Among these steps, how many were being done between t1 and t2? Same 8 steps | Number of steps |

Distractors are taken where possible from steps with no task-graph dependency on X, so the recipe
does not leak the time position.

## Not-done steps (hallucination)

- **Source:** Missing Step tags with no footage: 237 in the pool; 108 in the 64 main reports (47
  reports). These are real recipe steps, so only the video tells they were skipped.
- **One-step questions:** each adds "If the step was not done in the video, answer null." Any
  other answer to a not-done step is a hallucination; `null` on a filmed step is a false
  not-done.
- **8-step questions:** a not-done step may be an option; choosing or counting it is the
  hallucination.
- Asked in E1 only. The error row also tracks false alarms (a step with no mistake called wrong),
  `extras` and `extra_claims`.

## Supplementary content (outside the main matrix)

- **Order:** "Was X done before Y?", only for step pairs with no dependency and no time overlap.
  Median 13 pairs per report; pairs share steps, so they are not independent facts.
- **Skipped steps:** reuse `skipped_steps`.

## Deferred

- In the duration and localization rows, bool / multi / int ask a judgement ("longer than
  31.3 s", "within [t1, t2]") that is not the same fact as single / seconds. Ideally every step
  could be asked the same fact in every type, as in the error row.
