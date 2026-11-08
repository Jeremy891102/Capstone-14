# cc4d question count × answer-type mix test plan (draft)

## Goal

Keep the model fixed and vary only the **number of questions** in a report and the **share of
each answer type**, to see how a VLM's performance on questions about the same video changes.
The design follows `data/hd_epic_loc_nested_v1`: the video and the questions stay the same, and
only the subsets and their order change.

## Fixed conditions

- Model, generation settings (temperature 0), prompt template.
- Video input: the same `time_range`, fps and resolution.
- Context: `task_name`, `recipe`, `error_definitions`.
- Question text, options and answers are taken from `cc4d_dense_v1` unchanged.

## Why only three answer types

CaptainCook4D step annotations carry only two kinds of information: time (`start_time`,
`end_time`) and errors (`errors`). `correct` (bool), `error_types` (multi_choice) and
`duration` (seconds) already use both. A per-step `single_choice` or `int` question could only
come from one of two sources, and neither works:

- **Order**, e.g. "Which step was done right before X?" or "Which step number was X?". The
  `recipe` in the context gives these away. From the task graphs: the recipe alone fixes the
  position of 130 of 358 recipe steps (36%), and the remaining steps have a median of only 2
  possible positions. For a video where the steps were done correctly, the recipe text is enough
  to answer.
- **Duration**: it duplicates `duration`.


**To test all five answer types:** ask about the same duration in three formats.

| type | question | answer |
| --- | --- | --- |
| `seconds` | How many seconds did step "X" take? | `end_time − start_time` |
| `single_choice` | How long did step "X" take? (8 time ranges) | The range the duration falls in |
| `int` | How many seconds did step "X" take? (answer an integer) | The duration, rounded |

- Each step gets only one of the three. Otherwise the answers give each other away.
- The range boundaries are the octiles of the `cc4d_dense_v1` step durations, so the 8 ranges
  get roughly the same number of answers.
- `int` is scored by exact match. After rounding, that tolerance is much stricter than the
  max(2 s, 25%) of `seconds`. When comparing the two, remember the difference comes from the
  scoring rule, not from the format itself.
- Benefit: all three formats ask about the same thing, so the test measures the effect of the
  format alone. By contrast, the current bool, multi_choice and seconds questions each ask about
  something different, so the effect of the answer type cannot be separated from how hard the
  content is.

## Conditions (10 per video)

| questions | bool only | multi_choice only | seconds only | mixed (1/3 each) |
| --- | --- | --- | --- | --- |
| 6 | ✓ | ✓ | ✓ | ✓ 2/2/2 |
| 12 | ✓ | ✓ | ✓ | ✓ 4/4/4 |
| 24 | | | | ✓ 8/8/8 |
| 48 | | | | ✓ 16/16/16 |

- **Nested:** smaller sets are contained in larger ones, so the same question can be compared
  across conditions as a pair.
- **Steps in the mixed sets:** up to 16 questions, each step is asked only one type, because
  the `correct` and `error_types` answers of the same step can be inferred from each other. At
  24 and 48 questions, some steps must be asked more than one type; that is a limitation of
  those two conditions.
- **Order:** 3 orders per condition. Shuffle the 48 questions once, rotate by thirds, then filter
  out each condition's subset.
- **No per-question baseline:** asking each question on its own costs too much (every call sends
  the whole video). The 6-question sets serve as the low-interference end instead, so the results
  have no 1-question point.

## Calls

10 conditions × 3 orders = **30 calls** per report/video, not counting retries.

## Analysis

1. **Question-count effect:** accuracy of the mixed sets at 6 → 12 → 24 → 48, overall and per
   answer type.
2. **Answer-type effect:** at 6 and at 12 questions, compare whether the same question is right
   in the single-type set and in the mixed set, with a paired test (McNemar, or a logistic model
   with video and question as random effects).
3. **Position effect:** accuracy by where the question sits in the prompt (first, middle or last
   third).
4. Every result comes with the **majority-answer baseline** and the format failure counts
   (`missing_answer`, `invalid_answer`, `invalid_output`).

Scoring follows `docs/scoring_rules.md`.
