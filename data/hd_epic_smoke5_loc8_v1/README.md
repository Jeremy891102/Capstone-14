# Five-video localization smoke benchmark

Five reports, each referencing one original HD-EPIC MP4 and containing eight existing
single-choice localization questions. Whole-report and per-field use this same benchmark.
No new questions or reference answers were authored.

The first three reports retain the eight-question order-1 subsets from
`hd_epic_loc_nested_v1`. Two additional videos from different participants were selected
from `hd_epic_q40_v1` using `nested_selection(..., seed=20261007, sizes=(8,))`.
Original report windows are retained; question and option timestamps are clip-relative.
Each original question's input interval is fully inside its report window.

Files:

- `reports.jsonl`: five model-facing reports, 40 fields.
- `ground_truth.jsonl`: reference answers, used only for offline scoring.
- `question_review.md`: human-readable review copy with reference answers; never prompt input.
- `provenance.json`: seed, source file hashes and extraction provenance.

Media is stored outside Git. Set VIDEO_REPORT_VIDEO_ROOT to the directory containing
P09/, P02/, P04/, P01/ and P03/ original MP4s. Prepared clips are temporary local inputs.

Draft content-review status is retained: successful smoke execution does not independently
verify every source answer against the original video. Automated checks cover schema,
target mappings, local video availability, payload integrity and output parsing/scoring.

See docs/smoke_test_results.md for the completed five-video Gemini E2E results and evidence
links, and docs/smoke_test_logging.md for automatically generated logs.
