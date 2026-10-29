# CC4D answer-type experiment proposal (draft, 2026-10-29)

Chinese version: `docs/cc4d_answer_type_proposal.zh.md`. Both say the same thing.

| Research question | Experiments |
| :---- | :---- |
| Q1  Does accuracy change when a call asks more questions? Does the change depend on the answer type? | E1–E3; E5 extends to 16 |
| Q2  Does accuracy change when the same fact is asked in a different answer type? | E1 (rechecked in E3) |
| Q3  How does the model do on each content? | E1 |
| Q4  Does mixing answer types in one call affect the same questions? | E4 |
| Q5  Do the answer type and question count affect hallucination? Does the not-done step's position in the call matter? | E1–E3, E5 H batches |
| Q6  Does a question's position in the call affect accuracy? | Main k = 8 and E5 k = 16 batches, forward vs reversed |

- Models: Gemini 3.5 Flash-Lite first; once it passes the checks, the same questions on Gemini
  3.6 Flash
- Data: CaptainCook4D, a new benchmark built from the raw annotations
- Variables: question count (1 / 4 / 8 / 16), content (error / duration / localization), answer
  type, question order (forward / reversed)
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
  a hallucination; `null` on a filmed step is a false not-done.
- Tested at every question count: E1 asks one not-done step per question. For k > 1, every
  original batch gets an **H batch**: one fact is replaced by a not-done step from the same
  report, the other questions unchanged. The replaced position rotates across reports so every
  position from 1 to k is covered. Only reports with not-done steps get H batches.
- Analysis: hallucination rate by question count, by the question's position in the call (how
  far from the last question) and by the step's position in the recipe (how far from the end of
  the video). The other k − 1 questions of an H batch are paired with the original batch, to see
  whether adding a not-done step affects them.
- False alarms: an error-row step with no mistake called wrong. Recorded for every type and
  every question count.
- E4 gets no H batches, so the mixed-type comparison has a single variable.

**E Question order (Q6).** The largest-k batches (E3 at k = 8 in the main experiment, E5 at
k = 16, H batches included) run again with the question order reversed; option order stays the
same. A question sits at position p in one run and k + 1 − p in the other, compared item by item.
Position effects show most in the longest batches, so smaller k are not reversed. E4 is not
reversed either: each of its halves is already the target once.

**F Caching.** Each video is asked many times, so caching could save the video's input cost, but
it must first be shown not to change the answers. The pilot compares four conditions on existing
HD-EPIC questions: A no cache; B an exact repeat of A (measures the model's own instability); C
explicit cache; D no cache, with the prompt laid out as in C (video and question in separate user
turns). If C and D disagree no more than A and B, the main runs use caching; otherwise not.
Explicit caching currently needs a service account on the Vertex project or an AI Studio key;
without one, no caching. Either way every call logs its cached tokens, to check afterwards
whether calls that hit the cache answer differently.

## Experiments

| ID | Questions per call | Contents | Calls |
| :---- | :---- | :---- | :---- |
| E1 | 1 | 11 cells × 512 facts | 5,632 |
| E1 | 1 | 4 cells × 64 reports (8-step questions) | 256 |
| E1 | 1 | 108 not-done steps × 11 cells | 1,188 |
| E2 | 4 | 11 cells × 512 facts | 1,408 |
| E2-H | 4 | 11 cells × 47 reports × 2 batches | 1,034 |
| E3 | 8 | 11 cells × 512 facts | 704 |
| E3-H | 8 | 11 cells × 47 reports | 517 |
| E4 | 8 | 11 cells × 2 target halves × 64 reports | 1,408 |
| | | **Main subtotal** | **12,147** |
| E5 | 1 / 4 / 8 / 16 | Localization: 40 reports × 16 facts × 3 types | 2,760 |
| E5-H | 1 / 4 / 8 / 16 | Localization: about 20 not-done steps × 3 types (k = 1); about 12 reports × 3 types × 7 batches (k > 1) | about 312 |
| E5 | 1 / 4 / 8 / 16 | Duration: 44 reports × 16 facts × 3 types | 3,036 |
| E5-H | 1 / 4 / 8 / 16 | Duration: 19 not-done steps × 3 types (k = 1); 12 reports × 3 types × 7 batches (k > 1) | 309 |
| | | **Forward subtotal** | **about 18,564** |
| Reversed | 8 / 16 | Largest-k batches again in reversed order (main E3 + E3-H 1,221, E5 k = 16 about 324) | about 1,545 |
| | | **Total per model** | **about 20,109** |
| | | **Two models (3.5 Flash-Lite, 3.6 Flash)** | **about 40,218** |

The E5 localization H counts depend on how many not-done steps the 40 sampled reports contain;
the table estimates them by proportion.

About 5,200 video-hours per model, roughly 7.5 times the HD-EPIC baseline. Rough cost without
caching, from standard-tier input prices (2026-10-29): Flash-Lite about US$320, 3.6 Flash about
US$790 plus its thinking tokens. Caching would cut the input cost to roughly 10–20%. The pilot
measures the actual cost.


## Results to report

| Question | Figure | Numbers |
| :---- | :---- | :---- |
| Q1 | Per row, a line chart: X question count, Y accuracy, one line per type; plus cost and token charts. A separate set for E5 (k up to 16) | Paired change from k = 1; missing / invalid rates |
| Q2 | Per row, a paired dot plot: X type, Y accuracy, with chance and majority baselines | Paired differences with 95% intervals; chance-normalized accuracy; error row split into mistake / no mistake |
| Q3 | Bar chart: X content, Y chance-normalized accuracy | Descriptive |
| Q4 | Paired dot plot: single vs mixed, one panel per type | Paired change; counts of right→wrong and wrong→right |
| Q5 | Bar chart: X type, Y hallucination and false-alarm rates; line chart: X question count or position, Y hallucination rate | False not-done rate; `extras` / `extra_claims`; paired change of the other questions in H batches |
| Q6 | Line chart: X question position, Y accuracy, one line each for forward and reversed | Paired change, forward vs reversed |
| Models | Every figure above with Flash-Lite and 3.6 Flash side by side | Whether the two models reach the same conclusions |

## Common controls

- Fixed: model, temperature 0, prompt template, video (whole recording to 5 s after the last
  step, fps = 2), context (`task_name`, `recipe`, `error_definitions`), option order. Every call
  is independent.
- Missing / invalid count as wrong. The open_text judge is first checked against about 100
  hand-labelled answers.
- Every video weighs the same; bootstrap by video. Also recomputed with every recipe weighing the
  same.
- Flow: freeze IDs → pilot on 2 reports + cache test → check parsing, judge and cost →
  Flash-Lite runs E1–E5 → check results → 3.6 Flash runs the same set.

## Dataset preparation

1. **Reports:**
   - Main: recordings with at least 8 usable steps in every row and at least 4 steps with and 4
     without a mistake in the error row: 64, all used. 19 recipes (1–8 reports each); videos
     3.9–22.7 minutes; 47 reports have not-done steps, 108 in all.
   - E5: 125 reports have at least 16 filmed steps for localization (10 recipes, 9–16 each);
     4 are sampled per recipe, 40 in all, giving 640 facts per type (95% interval about ±3.9
     points). All 44 reports with at least 16 usable steps for duration are used (7 recipes; 12
     reports have not-done steps, 19 in all). May overlap with the main reports.
2. **Facts** (fixed seed):
   - Error row: 4 with a mistake, 4 without. The natural rate is only 21%, which leaves too few
     mistake questions, and answering "no mistake" everywhere would score 79%.
   - Duration row: 8 random steps, roughly half above and half below 31.3 s.
   - Localization row: 8 random filmed steps.
   - All not-done steps are included.
   - E5: 16 random steps per report per row; duration again roughly half above 31.3 s.
3. **Questions:** per `docs/cc4d_format_questions.md`. Options are generated once per fact and
   shared by every condition.
4. **Freeze:** fact order, E2/E3/E5 batches, the step and position swapped into each H batch,
   reversed batches, E4 target halves and filler rotation, question IDs, baselines for every
   cell.
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
- Position designs other than reversal (e.g. random shuffles): reversal already moves every
  question to its mirrored position.
- Reversal at smaller k, and of E4.
- The other 85 E5 localization reports.
