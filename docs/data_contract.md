# Data contract `video_report.v1`

This is the interface between **dataset preparation** (owned by a teammate; HD-EPIC,
CaptainCook4D, or others) and **this repository** (inference + evaluation). Prepared data that
follows this contract can be validated and run without touching source code.

The field names follow common evaluation conventions: Inspect AI's `Sample` uses `id`,
`input`, `choices`, `target` and `metadata`, and multiple-choice targets are letters. The
report wrapper around those fields is **our own project schema**. It is not an industry
standard.

Machine-checked definition: [`src/video_report/benchmarks/schema.py`](../src/video_report/benchmarks/schema.py).
Validator: `python scripts/validate_data.py --benchmark <dir> [--check-videos]`.

## Benchmark directory

```
benchmarks/<benchmark_id>/
├── manifest.json        # required
├── reports.jsonl        # required: model-facing reports (no answers!)
├── ground_truth.jsonl   # needed only for evaluation
└── mock_responses.jsonl # optional: canned answers for the mock provider
```

Real benchmark directories that contain dataset-derived content usually stay **outside Git**.
Only `benchmarks/mock_mcq_v1/` (synthetic) is committed.

### `manifest.json`

| key | type | notes |
|---|---|---|
| `schema_version` | `"video_report.v1"` | required, exact |
| `benchmark_id` | id | |
| `description` | str | |
| `reports_file` | str | default `reports.jsonl` |
| `ground_truth_file` | str \| null | default `ground_truth.jsonl` |
| `mock_responses_file` | str \| null | |
| `video_root` | str \| null | default root for relative video URIs, relative to the manifest dir |
| `source_datasets` | list[str] | provenance only |
| `notes` | str | |

Unknown keys are rejected.

## `reports.jsonl`: model-facing data

The file has one JSON object per line. **Everything in this file is allowed to influence a
request.** Only the parts listed below are actually rendered into one.

```json
{
  "id": "synthetic-0001",
  "input": {
    "videos": [
      {"id": "head_cam", "uri": "videos/synthetic_kitchen_0001.mp4", "mime_type": "video/mp4",
       "time_range": {"start": 12.0, "end": 48.5, "unit": "seconds", "reference": "video_start"}}
    ],
    "context": {"task_name": "Make a cup of tea", "recipe_step_count": 5}
  },
  "fields": [
    {"id": "primary_appliance", "question": "Which appliance does the person use to heat the water?",
     "answer_type": "single_choice",
     "choices": ["Electric kettle", "Microwave", "Pot on the stove", "Coffee machine"]}
  ],
  "source": {"dataset": "synthetic", "split": "test",
             "original_ids": {"synthetic_id": "0001"},
             "annotation_refs": ["synthetic://annotations/0001"]},
  "metadata": {"created_by": "fixture"}
}
```

| path | rules |
|---|---|
| `id` | **stable** report id, unique in the file. 1–128 chars `[A-Za-z0-9._-]`, starting alphanumeric |
| `input.videos[]` | ≥ 1; video `id`s unique within the report |
| `input.videos[].uri` | relative path, absolute path, or URI with scheme (see below) |
| `input.videos[].mime_type` | optional; inferred from extension for local files |
| `input.videos[].time_range` | optional, see *Time ranges* |
| `input.context` | optional `{str: str\|int\|float\|bool}`. **Candidates** for model context: a key is rendered only if the experiment whitelists it in `prompt.context_keys` |
| `fields[]` | ≥ 1; field `id`s unique within the report, stable across dataset versions |
| `fields[].question` | non-empty |
| `fields[].answer_type` | `single_choice`, `multi_choice`, `bool`, `int` or `seconds`. Anything else is rejected at load time. **Only `single_choice` is scored so far**; `evaluate` refuses runs with other types |
| `fields[].choices` | `single_choice` / `multi_choice`: ordered list of 2–702 unique non-empty strings, labelled A..Z, then AA..AZ, BA..ZZ (choice 26 is `AA`). Other types: omit (must be empty) |
| `fields[].allow_not_visible` | optional, default `false`. The model may answer `"not_visible"` (hallucination checks). Set it on every field of a report, not only the trap fields, or its presence gives the answer away |
| `source` | provenance (`dataset`, `split`, `original_ids`, `annotation_refs`). **Never sent to the model** |
| `metadata` | free-form bookkeeping. **Never sent to the model** |

Unknown keys are rejected everywhere, which works as a leak guard. A `target`, `answer` or
`evidence` key in this file fails validation instead of reaching a prompt.

### Time ranges

```json
{"start": 12.0, "end": 48.5, "unit": "seconds", "reference": "video_start"}
```

* The interval is **half-open, `[start, end)`**, with `0 ≤ start < end` and both values finite.
* `unit` must be `seconds` in v1. Converters must turn frames into seconds themselves, using
  the source video's real frame rate.
* `reference` must be `video_start`. Offsets are measured from the first frame of the
  referenced file (presentation time 0). They are not dataset-global or wall-clock times. If
  a dataset's annotations refer to a longer recording than the file you ship, convert them.
* Whether a time range is used is decided per experiment (`video.use_time_range`). The Gemini
  adapter sends it as `VideoMetadata(start_offset, end_offset)`. The mock provider ignores
  videos.

### Video path resolution

The `uri` field takes one of these forms:

| form | example | resolved as |
|---|---|---|
| relative path | `P01/P01_01.mp4` | `<video root>/P01/P01_01.mp4` |
| absolute path | `/data/hd-epic/P01_01.mp4` | as is |
| `file://` URI | `file:///data/x.mp4` | local path |
| other scheme | `gs://…`, `https://…` | kept as a remote URI. **The Gemini adapter v1 rejects these** |

The **video root** is taken from the first of these that is set:

1. `--video-root DIR` on the command line
2. environment variable `VIDEO_REPORT_VIDEO_ROOT`
3. `data.video_root` in the experiment config (relative to the config file)
4. `video_root` in `manifest.json` (relative to the manifest)
5. the benchmark directory itself

Recommended setup: reports use relative URIs, and each teammate exports
`VIDEO_REPORT_VIDEO_ROOT=/path/to/their/videos`. No source or config edits are needed.

Resolved absolute paths are frozen into each run's `snapshot/requests.jsonl`. This makes a run
directory specific to one machine. To change a video location, start a new run.

## `ground_truth.jsonl`: evaluation-only

```json
{"report_id": "synthetic-0001", "field_id": "primary_appliance", "target": "A",
 "target_text": "Electric kettle",
 "evidence": {"annotation": "kettle switched on at 14.2s", "annotator": "synthetic"}}
```

| key | rules |
|---|---|
| `report_id`, `field_id` | must reference an existing report/field; a `(report_id, field_id)` pair may appear only once |
| `target` | by `answer_type`: `single_choice` → one choice label (`A`, …, `Z`, `AA`, …); `multi_choice` → list of distinct labels (`[]` = none); `bool` → `true`/`false`; `int` → integer; `seconds` → number ≥ 0. With `allow_not_visible`, any type may instead be `"not_visible"` |
| `target_text` | optional, `single_choice` letter targets only: must equal `choices[target]` exactly (catches choice reordering) |
| `evidence` | privileged annotation evidence. Read only by evaluation, never by methods, prompt builders or providers |

Inference never reads this file. Evaluation requires a valid target for **every selected
field** of the run and fails otherwise. Targets for reports or fields outside the run's
selection are ignored.

## `mock_responses.jsonl` (optional, mock provider only)

```json
{"report_id": "synthetic-0001",
 "answers": {"primary_appliance": "A", "first_item_in_cup": "C", "task_outcome": "C"},
 "raw_by_call": {"synthetic-0001::field=task_outcome": "The person finishes making tea. Answer: C"}}
```

These are canned answers, written independently of the ground truth. A `null` answer leaves
that key out of the response. `raw_by_call` returns exact raw text for one call id, which is
useful for exercising parse failures.

## Integrating real data (teammate checklist)

1. Produce `manifest.json`, `reports.jsonl`, and `ground_truth.jsonl` in a directory outside
   Git, following the rules above. Keep dataset-specific conversion in the data-prep code, not
   here.
2. Put privileged or derived information (annotation text, timestamps of evidence, annotator
   ids) only in `ground_truth.jsonl` → `evidence`, or in `source`/`metadata`. Model-visible
   hints go in `input.context`, and only matter once an experiment whitelists them.
3. `python scripts/validate_data.py --benchmark /path/to/bench --check-videos --video-root /path/to/videos`
4. Copy an experiment directory, point `data.benchmark` at the new directory, and optionally
   restrict `data.report_ids` / `data.field_ids` for a small smoke test.
5. Run with the mock provider first (write a `mock_responses.jsonl` or skip this step), then
   with Gemini.

If the prepared format differs slightly (for example one file per report), write a small
loader that yields plain dicts and validate each one with
`video_report.datasets.base.parse_report` / `parse_target`. That keeps all validation in one
place.

## Versioning

Any incompatible change to these rules gets a new `schema_version`. A version-1 reader rejects
data from any other version.
