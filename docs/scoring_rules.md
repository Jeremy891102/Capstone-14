# Scoring rules by answer type

How each answer type is judged, from parsed answer to metrics. Field and target formats
are defined in `docs/data_contract.md`.

**Status (2026-10-08):**

- Only `single_choice` is implemented, in `evaluation/scorers.py`. The other types and
  `not_visible` are decided here, but no code exists yet. Until it does, `video-report
  evaluate` refuses runs that contain them.
- The rules below were agreed for the CaptainCook4D benchmark (`cc4d_dense_v1`). They apply
  to any benchmark that uses these types.

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

## Rules per type

### `single_choice` (implemented)

- **Valid answer:** one choice label within the field's choices: `"A"` … `"Z"`, then
  `"AA"`, `"AB"`, ….
- **Correct:** the label equals the target.

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

### `multi_choice`

- **Valid answer:** a JSON list of distinct labels, each within the field's choices. Order
  does not matter. `[]` means "none".
  - A list with duplicate or unknown labels is invalid.
- **Correct:** the set of answered labels is exactly the set of target labels. When the
  target is `[]`, only `[]` is correct.
- **Also recorded** for every valid answer:

  | count | meaning | example: target `["A", "C"]`, answer `["C", "D"]` |
  | --- | --- | --- |
  | `hits` | Labels in both the target and the answer | 1 (C) |
  | `misses` | Labels in the target that were not answered | 1 (A) |
  | `extras` | Labels answered that are not in the target | 1 (D) |

  Precision and recall can be derived later: `hits / (hits + extras)` and
  `hits / (hits + misses)`.

### `not_visible` (any type with `allow_not_visible: true`)

- **Valid answer:** the exact string `"not_visible"`. Any other spelling or case is invalid.
  A field without `allow_not_visible` treats it as invalid.
- **Primary outcome:**

  | target | answer | primary outcome | also tagged as |
  | --- | --- | --- | --- |
  | `not_visible` | `not_visible` | correct | |
  | `not_visible` | any valid concrete answer | incorrect | `hallucination` |
  | concrete value | `not_visible` | incorrect | `over_abstention` |
  | concrete value | concrete value | by the type's rule | |

- For a `multi_choice` field, `[]` is a concrete answer ("no mistakes"), not
  `not_visible`.

## Metrics to report

1. **Field accuracy:** overall, per answer type, and per field kind. Examples of field kinds:
   `stepNN_correct`, `stepNN_duration`, `skipped_steps`. For CC4D, a field kind is the field
   id with the `stepNN_` prefix removed.
2. **Hallucination rate:** among fields whose target is `not_visible`, the share answered
   with a concrete value.
3. **Over-abstention rate:** among fields with a concrete target, the share answered
   `not_visible`.

   Always report 2 and 3 together. Answering `not_visible` everywhere would give a
   hallucination rate of 0.
4. **Majority baseline per field kind:** the accuracy of always giving the most common
   target. Report it next to the model's accuracy. Example: on CC4D, `stepNN_correct =
   true` alone gets 65.5% on real steps.
5. **Secondary measurements:**
   - `int` / `seconds`: median `abs_error`, plus median `rel_error` for `seconds`.
   - `multi_choice`: total `hits` / `misses` / `extras`.

   Computed only over valid, concrete answers.
6. **Format failures:** `invalid_answer` / `missing_answer` / `invalid_output` counts, per
   type.
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
