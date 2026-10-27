# Scoring rules by answer type

How each answer type is judged, from parsed answer to metrics. Field and target formats
are defined in `docs/data_contract.md`.

**Status (2026-10-13):**

- Only `single_choice` is implemented, in `evaluation/scorers.py`. The other types
  are decided here, but no code exists yet. Until it does, `video-report
  evaluate` refuses runs that contain them.
- The rules below were agreed while designing the CaptainCook4D benchmark (`cc4d_dense_v1`).
  They apply to any benchmark that uses these types.
- `cc4d_dense_v1` uses `bool`, `int`, `seconds`, `single_choice` and `multi_choice`.
- 2026-10-26 additions for the CC4D answer-type experiment (`docs/cc4d_format_questions.md`):
  `open_text`, `single_choice` with several acceptable labels, `seconds` timestamps, and
  cross-type comparison (metric 6).

## Principles

1. **Every field gets one primary outcome: correct or not.** Overall accuracy averages it
   across all types. The denominator is every selected field, and missing, invalid and
   not-executed answers count as wrong, the same as today.
2. **Some types also record secondary measurements** (errors, option counts). These are for
   analysis only and never change the primary outcome.
3. **Answers are checked against their JSON type, strictly.** A value of the wrong type or
   shape is `invalid_answer`, never coerced. Formatting failures stay separate from wrong
   answers. Examples: the string `"true"` for a bool, `"30 s"` for seconds.

## Outcomes

These six outcomes apply to every type and are unchanged from the current scorer:

| outcome | meaning |
| --- | --- |
| `correct` | Valid answer that the type's rule accepts |
| `incorrect` | Valid answer that the type's rule rejects |
| `invalid_answer` | The key is present, but the value is not a valid answer for this field's type |
| `missing_answer` | The response parsed, but this field's key is absent |
| `invalid_output` | The whole response is unusable (empty, invalid JSON, duplicate keys, not an object) |
| `not_executed` | No successful provider call covered this field |

## Not-done steps (`null`)

Decided 2026-10-26 for the CC4D answer-type experiment. A field about one step may allow
`null`, meaning "this step was not done in the video". The question says so in the same
words for every type.

- **Valid answer:** `null` is valid for every type when the field allows it.
- **Correct:** `null` is correct only when the target is `null` (a Missing Step with no
  footage). Any other answer to such a field is incorrect: a **hallucination**.
- `null` on a filmed step is incorrect, recorded as a **false not-done**.
- Report both rates on their own, next to accuracy.
- Fields that list several steps as options (CC4D duration and localization
  `multi_choice` / `int`) do not use `null`: a not-done step can appear as an option, and
  selecting or counting it is the hallucination.

## Rules per type

### `single_choice` (implemented)

- **Valid answer:** one choice label within the field's choices: `"A"` … `"Z"`, then
  `"AA"`, `"AB"`, ….
- **Correct:** the label equals the target.
- **Several acceptable labels** (decided 2026-10-26): the target may be a list of labels,
  and any one of them is correct. Used by the CC4D error-category question, where a step
  with two mistakes accepts either. The question text must say "pick any one". The option
  list stays the same for every question, so the format is not changed by the answer.

### `bool`

- **Valid answer:** JSON `true` or `false`.
- **Correct:** equal to the target.

### `int`

- **Valid answer:** a JSON integer. `true`/`false` and `3.0` are invalid.
- **Correct:** exactly equal to the target. No tolerance, not even ±1 (decided 2026-10-08).
- **Also recorded:** `abs_error = |answer − target|`.

### `seconds`

- **Valid answer:** a finite JSON number ≥ 0, integer or decimal.
- **Correct:** `|answer − target| ≤ max(2, 0.25 × target)`. The tolerance is computed from
  the **target**, not the answer.
  - Below 8 s, the 2 s floor applies. It roughly covers annotation boundary noise.
  - From 8 s up, the 25% applies.

  | target | accepted range |
  | --- | --- |
  | 4 s | 2–6 s |
  | 8 s | 6–10 s |
  | 40 s | 30–50 s |
  | 120 s | 90–150 s |

- **Also recorded:** `abs_error = |answer − target|`, and `rel_error = abs_error / target`
  (null when target is 0).
- **Configurable:** both tolerance values (floor 2 s, ratio 0.25) live in the scoring
  config, and each evaluation records the values it used. Changing them only needs a
  re-evaluation, not a new run.
- **Timestamps** (decided 2026-10-26, e.g. "At what time did step X start?"): the tolerance
  above would grow with the timestamp (±150 s at 600 s), so a timestamp field instead takes
  its tolerance from the step's own length: correct when
  `|answer − start| ≤ max(2, 0.25 × (end − start))`. Same floor and ratio as durations. The
  field marks itself as a timestamp and carries the step length in its ground truth.

### `multi_choice`

- **Valid answer:** a JSON list of distinct labels, each within the field's choices. Order
  does not matter. `[]` means "none".
  - A list with duplicate or unknown labels is invalid.
- **Correct:** the set of answered labels is exactly the set of target labels. When the
  target is `[]`, only `[]` is correct.
- **Exclusive option** (decided 2026-10-26): a field may mark one choice as exclusive, such as
  CC4D's "No mistake", so the option list matches the `single_choice` version of the same
  question. That option may only be answered alone: `["F"]` is valid, `["F", "A"]` is
  `invalid_answer`. The target for "none" is then `["F"]` and `[]` is `invalid_answer`.
  For hits / misses / extras, the exclusive label counts like any other.
- **Also recorded** for every valid answer:

  | count | meaning | example: target `["A", "C"]`, answer `["C", "D"]` |
  | --- | --- | --- |
  | `hits` | Labels in both the target and the answer | 1 (C) |
  | `misses` | Labels in the target that were not answered | 1 (A) |
  | `extras` | Labels answered that are not in the target | 1 (D) |

  Precision and recall can be derived later: `hits / (hits + extras)` and
  `hits / (hits + misses)`.

### `open_text`

Free-text answers judged by an LLM against a short reference. Decided 2026-10-26 for the
CC4D per-step question "Compared with the recipe, what did the person do differently in the
step "…"? Answer "none" if the step was done as written."

- **Valid answer:** a non-empty JSON string.
- **Judge input** (text only, no video): the step description, the reference (a list of error
  descriptions, or `none`), the step's `modified_description` when it exists, and the answer.
- **Correct:**
  1. Reference `none`: correct only if the answer says nothing was done differently. Naming
     any mistake is incorrect.
  2. Reference has errors: correct if the answer names **at least one** reference error by
     what was actually done wrong. Naming only the error category, or a vague "done badly",
     does not count. Different wording with the same meaning counts.
  3. Extra claimed mistakes do not change the outcome.
- **Judge output:** `{"correct": bool, "matched": [reference indices], "extra_claims": int}`.
- **Also recorded:** `matched` and `extra_claims` (the counterpart of `multi_choice`
  `extras`).
- **Configurable:** the judge model, the judge prompt version, and the match rule ("at least
  one" vs "all") live in the scoring config, and each evaluation records them. Changing them
  only needs a re-evaluation, not a new run.
- **Validation:** before use, label about 100 answers by hand and report agreement with the
  judge.

**CC4D question pool** (from the 5,295 steps of `cc4d_dense_v1`):

| steps | count | handling |
| --- | --- | --- |
| filmed, no errors | 3,591 | asked, reference `none` |
| filmed, at least one Technique / Preparation / Measurement / Timing / Temperature error | 1,118 | asked, reference = the descriptions of those errors only (Order descriptions dropped) |
| filmed, only Order errors | 334 | not asked (order needs the whole video, not the step) |
| Other | 6 | not asked |
| Missing Step, no footage | 237 | asked as hallucination items, target `null` (see "Not-done steps" above) |
| Missing Step but has footage | 3 | not asked (contradictory annotation) |
| no footage, no Missing Step tag | 6 | not asked (optional step "Top with more parmesan if desired") |

So 4,709 filmed steps, minus the 13 manual exclusions below: 4,696. Answering `none`
everywhere scores 76.5% (3,591 / 4,696). The same 4,696 steps are the pool for every
error-question type, so they can be paired (`docs/cc4d_format_questions.md`).

Manual exclusions: references too vague to judge, found by screening descriptions with ≤ 3
words or generic wording ("improperly", "wrong") and reading them by hand. The screen is not
exhaustive.

| recording | step_id | reference |
| --- | --- | --- |
| 1_34 | 9 | The top piece |
| 2_28 | 17 | Not properly peeling the onion. |
| 2_30 | 17 | Peel improperly |
| 2_46 | 14 | Peeled the garlic improperly |
| 5_28 | 71 | The water temperature is wrong |
| 5_42 | 60 | Not properly placing the paper filter. |
| 16_40 | 176 | Not properly peeling the Garlic cloves. |
| 16_40 | 180 | Not properly peeling the Garlic cloves. |
| 18_28 | 200 | Peel improperly |
| 20_26 | 176 | Peel garlic cloves improperly |
| 20_39 | 176 | Improperly peel the garlic cloves. |
| 26_24 | 303 | Used wrong shaped mug |
| 28_50 | 330 | Rolling the lettuce wraps in a wrong way. |

## Metrics to report

1. **Field accuracy:** overall, per answer type, and per field kind. Examples of field kinds:
   `stepNN_correct`, `stepNN_duration`, `skipped_steps`. For CC4D, a field kind is the field
   id with the `stepNN_` prefix removed.
2. **Majority baseline per field kind:** the accuracy of always giving the most common
   target. Report it next to the model's accuracy. Example: on CC4D, `stepNN_correct =
   true` alone gets about two thirds on real steps (the exact figure is in the dataset
   README).
3. **Secondary measurements:**
   - `int` / `seconds`: median `abs_error`, plus median `rel_error` for `seconds`.
   - `multi_choice`: total `hits` / `misses` / `extras`.
   - `open_text`: total `extra_claims`, and judge agreement from the hand-labelled sample.

   Computed only over valid, concrete answers.
4. **Skipped-step misses (the hallucination check for `cc4d_dense_v1`):** `cc4d_dense_v1`
   has no decoy steps, so the check is `skipped_steps`. A target label the model did not
   answer is a step that was not done but that the model treated as done. Report these
   `misses` and the share `misses / (hits + misses)` on their own, separately from `extras`
   (steps that were done but called skipped). Computed only over valid answers.
5. **Format failures:** `invalid_answer` / `missing_answer` / `invalid_output` counts, per
   type.
6. **Comparing accuracy across types.** Raw accuracy is not comparable between types,
   because their chance levels differ widely: `bool` 50%, 8-option `single_choice` 12.5%,
   exact match on a 7-option `multi_choice` close to 0, and `int` / `seconds` depending on
   the tolerance. So:
   - Every accuracy cell (type × content × condition) reports its **chance baseline** and its
     **majority baseline** (item 2) next to it.
   - A type effect is measured as a **paired difference**: the same underlying fact, asked in
     different types in separate calls, compared item by item.
   - When cells must be compared directly, also report chance-normalized accuracy:
     `(acc − chance) / (1 − chance)`. For `int` / `seconds`, chance is the hit rate of
     always answering the majority-baseline value under the same tolerance.
7. **Per-option breakdown for `multi_choice` fields with a fixed option set** (CC4D
   `stepNN_error_types`, whose options are the 7 error tags). For each option, report:
   - **recall:** among fields whose target contains the option, the share whose answer also
     contains it. This measures missed errors.
   - **precision:** among fields whose answer contains the option, the share whose target
     also contains it. This measures false alarms.
   - the counts behind both numbers.

   Use only valid, concrete answers. Example: a recall of 20% on `Order` and 70% on
   `Technique` means the model sees spills but not sequencing mistakes. The summed
   `hits` / `misses` / `extras` would hide that. Fields whose options vary per report, such
   as `skipped_steps`, get only the summed counts.
