# CC4D Answer-Type Experiment: Results Overview

Design: [cc4d_answer_type_proposal.md](../cc4d_answer_type_proposal.md), [cc4d_format_questions.md](../cc4d_format_questions.md). This page gives the main conclusion per research question; details, tables and figures are in the linked files.
## Data and models

| Model | Coverage | Note |
| --- | --- | --- |
| Gemini 3.5 Flash-Lite | Full (41 groups, 25,711 calls) | Mostly near chance |
| Qwen3-VL-32B | Full (same groups); open_text scored 11/8 | Long-video resolution reduced |
| Gemini 4 Argon | Base group only, 5 cells (duration bool/single; localization bool/single/seconds); k = 1/4/8 and reversed k = 8 | No error content, E4, hallucination items or E5 |

Raw results: `results/cc4d_answer_types_v1_*`. Analysis scripts and outputs: `results/cc4d_q1_*`, `cc4d_q2.py` to `cc4d_q6.py`, `*_out.txt`. Figures: [figs/](figs/).

## Main conclusions

| Question | Conclusion | Details |
| --- | --- | --- |
| **Q1** Does asking more questions per call change accuracy? Does it depend on answer type? | **Depends on the model, not the type.** Qwen drops with k (k = 1 → 8: about −4.6 pt), mostly on localization bool/single; at k = 16 localization falls about 10 pt. Flash-Lite and Argon show no reliable effect. The k effect varies by content (3-way interaction p = 0.0002) but not by type (duration + localization, p = 0.85). Part of Qwen's drop is a growing "yes" bias on bool. | [q1.md](q1.md) |
| **Q2** Does asking the same fact in a different answer type change accuracy? | **Yes, but mostly answer bias, not ability.** In error content, accuracy comes almost entirely from no-mistake steps; mistake steps are answered correctly <33% in every type. Type rankings differ by model (model × type p < 0.0001); common ground: localization single beats bool, duration seconds is worse than bool. | [q2.md](q2.md) |
| **Q3** How does each model do on each content? | **Qwen and Argon: localization > duration > error. Flash-Lite is near chance on all three.** Balanced bool accuracy confirms real discrimination. Descriptive only: the three contents use different facts. | [q3.md](q3.md) |
| **Q4** Does mixing answer types in one call affect the same questions? | **No harm; a small gain** (Flash-Lite and Qwen both about +2 pt), concentrated in bool, consistent across models. Position-matched pairs are the clean test; moving a target later costs Qwen 4 pt. Cause of the gain unknown; not a change in answer bias. | [q4.md](q4.md) |
| **Q5** Do answer type and k affect hallucination? Does the position of a not-done step matter? | **Type matters a lot, with different rankings per model; k acts in opposite directions.** Flash-Lite hallucinates less (66% → 50% → 52%) but more often calls filmed steps not visible (2.9% → 6.4%); Qwen rises slightly (70% → 74% → 77%). No clear position effect; adding a not-done step changes other questions by ≤ 1.1 pt. | [q5.md](q5.md) |
| **Q6** Does question order/position affect accuracy? | **Only Qwen: reversed order hurts, more so at larger k** (k = 8: about −4 pt; k = 16 localization: about −19 pt). It is reversal itself, not "later is worse". Flash-Lite and Argon are unaffected. | [q6.md](q6.md) |

## Cross-question findings

1. **Answer bias is more common than ability differences.** Both Flash-Lite and Qwen lean toward "no mistake" in error content (Q2); Qwen's bool answers drift toward "yes" as k grows (Q1); Qwen's open_text gain with k comes from answering "none" more often (Q1). Flash-Lite's int gain is probably the same thing (unverified). Always split accuracy by mistake/no-mistake and by true answer.
2. **Qwen is the most sensitive to call format.** More questions (Q1), reversed order (Q6) and later position (Q4) all hurt it, concentrated in localization and duration, its strongest contents. Neither Gemini model shows a reliable k or order effect in the tested range.
3. **Mixing types does not hurt** (Q4), so mixed-type calls are safe for Flash-Lite and Qwen.
4. **Flash-Lite lowers hallucination by being more conservative** (Q5): at higher k it says "not visible" and "no mistake" more often, at the cost of more false not-done answers.
5. **Content difficulty (Q3) overlaps with where Qwen drops (Q1, localization)**, so k/order effects and content difficulty cannot be cleanly separated.

## Shared limitations

- **Argon has only 5 cells.** Its "no k/order effect" holds for duration and localization only, and its localization is near ceiling (95–97%), limiting sensitivity.
- **Flash-Lite is mostly near chance;** no detected effect is not proof of none.
- **No multiple-comparison correction;** treat p between 0.01 and 0.05 as a lead.
- **Contents and types are not fully crossed,** so analyses are run in blocks, not one factorial model; content comparisons are descriptive (different facts).
- **Qwen's long-video resolution was reduced,** so it is not the same condition as the other models.
- **Some scoring rules were adjusted post hoc:** Flash-Lite's `typed_v3` (e.g. reading 1004 as 10:04) was fixed after seeing results; the open_text judge agrees with 100 human labels 78% of the time and is strict.
- **Some causes remain open:** the Q4 bool gain, and Qwen's reversed-order drop (only partly traced to off-by-one answer misalignment).
