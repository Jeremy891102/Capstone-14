# CC4D answer-type experiment proposal (draft, 2026-10-26)

Chinese version: `docs/cc4d_answer_type_proposal.zh.md`. Both say the same thing.

| Research question | Experiments |
| :---- | :---- |
| Q1  Does accuracy change when a call asks more questions? Does the change depend on the answer type? | E1–E3; E5 extends to 16 |
| Q2  Does accuracy change when the same fact is asked in a different answer type? | E1 (rechecked in E3) |
| Q3  How does the model do on each content? | E1 |
| Q4  Does mixing answer types in one call affect the same questions? | E4 |
| Q5  Does the answer type affect hallucination? | E1–E3 |

- Model: Gemini 3.5 Flash-Lite
- Data: CaptainCook4D, a new benchmark built from the raw annotations
- Variables: question count (1 / 4 / 8), content (error / duration / localization), answer type
- Question rules: `docs/cc4d_format_questions.md`. Scoring rules: `docs/scoring_rules.md`

## Cells

A **fact** is one statement the annotations make about one step, such as "step X has no
mistake". Each fact is asked once in every answer type of its row, each time in a separate call.

| Content \ type | bool | single | multi | seconds | int (count) | open_text |
| --- | --- | --- | --- | --- | --- | --- |
| Error | one step | one step | one step | — | one step | one step |
| Duration | one step | one step | 8 steps | one step | 8 steps | — |
| Localization | one step | one step | 8 steps | one step | 8 steps | — |

"One step" means one question asks about one step: 11 cells, used in every experiment. "8 steps"
means one question lists 8 steps: 4 cells. Each report has only enough facts per row for one
such question, so these cells appear in E1 only.

## Design

Each report has 8 fixed facts per row, F1–F8.

**A Question count (Q1).** The same 8 facts are asked three ways. Each call has one content and
one answer type:

| Experiment | Questions per call | Calls for 8 facts |
| --- | --- | --- |
| E1 | 1 | 8 |
| E2 | 4 | 2 |
| E3 | 8 | 1 |

The main experiment stops at 8 because a step gives only one question per row. Requiring 16 in
all three rows leaves 37 reports, with only 5% error steps.

**A2 Question count up to 16 (E5, Q1).** HD-EPIC showed no count effect below 16, so the
localization and duration rows are extended to 16. Each row uses its own reports (at least 16
usable steps in that row), with 16 fixed facts per report, asked at k = 1 / 4 / 8 / 16, in bool,
single and seconds only. These reports have longer videos and fewer recipes, so E5 is compared
only within itself and is not joined to the main curve. The error row is left out: 16 questions
would hold only 1–2 mistakes on average.

**B Answer type and content (Q2, Q3).** Uses E1. Answer types are compared item by item, paired.
Contents are compared only within bool and single, the only types present in all three rows.
This comparison is descriptive.

**C Mixed answer types (Q4).** Four target questions use type T; the four filler questions use
the row's other types. Only the targets are scored, paired with the same questions in E3.

| Position | E3 (single type) | E4 (mixed), error row, T = bool |
| --- | --- | --- |
| 1–4 | F1–F4 bool (targets) | F1–F4 bool (targets) |
| 5–8 | F5–F8 bool | F5 single, F6 multi, F7 int, F8 open_text |

In the error row the fillers are one question each of the other 4 types; in the duration and
localization rows, two each of the other 2 types. The filler type assignment rotates across
reports. Each cell runs twice (F1–F4 as targets once, F5–F8 once); the same E3 call, scored in
halves, is the baseline.

**D Hallucination (Q5).**
- Not-done steps: every question allows the answer `null`. Any other answer to a not-done step is
  a hallucination; `null` on a filmed step is a false not-done. Asked in E1 only, so E2–E4 use
  the same filmed facts.
- False alarms: an error-row step with no mistake called wrong. Recorded for every type and
  every question count.

## Experiments

| ID | Questions per call | Contents | Calls |
| :---- | :---- | :---- | :---- |
| E1 | 1 | 11 cells × 512 facts | 5,632 |
| E1 | 1 | 4 cells × 64 reports (8-step questions) | 256 |
| E1 | 1 | 108 not-done steps × 11 cells | 1,188 |
| E2 | 4 | 11 cells × 512 facts | 1,408 |
| E3 | 8 | 11 cells × 512 facts | 704 |
| E4 | 8 | 11 cells × 2 target halves × 64 reports | 1,408 |
| | | **Main subtotal** | **10,596** |
| E5 | 1 / 4 / 8 / 16 | Localization: 125 reports × 16 facts × 3 types | 8,625 |
| E5 | 1 / 4 / 8 / 16 | Duration: 44 reports × 16 facts × 3 types | 3,036 |
| | | **Total** | **22,257** |

Main-experiment videos average 12.5 minutes, about 2,210 hours across all calls. E5 videos are
longer (about 20–24 minutes), about 4,100 hours. About 6,300 hours in all, roughly 9 times the
HD-EPIC baseline (about 690 hours). Actual cost and run time come from the pilot. With 512 facts
per row in the main experiment, the 95% interval near 50% accuracy is about ±4.3 points.

## Results to report

| Question | Figure | Numbers |
| :---- | :---- | :---- |
| Q1 | Per row, a line chart: X question count, Y accuracy, one line per type; plus cost and token charts. A separate set for E5 (k up to 16) | Paired change from k = 1; missing / invalid rates |
| Q2 | Per row, a paired dot plot: X type, Y accuracy, with chance and majority baselines | Paired differences with 95% intervals; chance-normalized accuracy; error row split into mistake / no mistake |
| Q3 | Bar chart: X content, Y chance-normalized accuracy | Descriptive |
| Q4 | Paired dot plot: single vs mixed, one panel per type | Paired change; counts of right→wrong and wrong→right |
| Q5 | Bar chart: X type, Y hallucination and false-alarm rates | False not-done rate; `extras` / `extra_claims` |

## Common controls

- Fixed: model, temperature 0, prompt template, video (whole recording to 5 s after the last
  step, fps = 2), context (`task_name`, `recipe`, `error_definitions`), option order. Every call
  is independent.
- Missing / invalid count as wrong. The open_text judge is first checked against about 100
  hand-labelled answers.
- Every video weighs the same; bootstrap by video. Also recomputed with every recipe weighing the
  same.
- Flow: freeze IDs → pilot on 2 reports → check parsing, judge and cost → run E1–E5.

## Dataset preparation

1. **Reports:**
   - Main: recordings with at least 8 usable steps in every row and at least 4 steps with and 4
     without a mistake in the error row: 64, all used. 19 recipes (1–8 reports each); videos
     3.9–22.7 minutes; 47 reports have not-done steps, 108 in all.
   - E5: 125 reports with at least 16 filmed steps for localization (10 recipes); 44 reports with
     at least 16 usable steps for duration (7 recipes). May overlap with the main reports.
2. **Facts** (fixed seed):
   - Error row: 4 with a mistake, 4 without. The natural rate is only 21%, which leaves too few
     mistake questions, and answering "no mistake" everywhere would score 79%.
   - Duration row: 8 random steps, roughly half above and half below 31.3 s.
   - Localization row: 8 random filmed steps.
   - All not-done steps are included.
   - E5: 16 random steps per report per row; duration again roughly half above 31.3 s.
3. **Questions:** per `docs/cc4d_format_questions.md`. Options are generated once per fact and
   shared by every condition.
4. **Freeze:** fact order, E2/E3/E5 batches, E4 target halves and filler rotation, question IDs,
   baselines for every cell.
5. **Validate:** `video-report validate-data`, leak checks, and a sample check that questions can
   be answered from the footage.

## Deferred

- More than 8 questions in the error row: too few mistakes; any ratio other than 4/4 only
  measures false alarms.
- More than 16 questions: 24 localization questions leave 12 reports, all one recipe.
- Mixed content: confounded with type mixing, and the same step leaks across questions.
- Turning duration and localization multi / int into one-step questions.
- E4b, single filler type: all 4 fillers in one type, to find which type affects the targets;
  6,720 more calls.
- Order and skipped-step questions.
