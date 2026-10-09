# Capstone-14: video → structured report, inference & evaluation

This is a small, testable harness for experiments that turn **egocentric task videos** into
**structured reports**: several single-choice fields per video. The first comparison is
**one model call for the whole report** against **one call per field**.

* Mock provider: fully offline, no SDK or API key needed.
* Gemini provider: minimal video path through the official `google-genai` SDK. **Not yet
  validated against the live API** (see [Status](#status)).
* Each run is frozen, resumable, and evaluated offline in a separate step.

Docs:
* [docs/multi_provider_pilot.md](docs/multi_provider_pilot.md): OpenAI, Dennis's Vertex Gemini,
  local key setup, and the single-type nested localization pilot (offline-verified only).
* [docs/openrouter_response_fixes.md](docs/openrouter_response_fixes.md): why OpenRouter token
  accounting and response validation were corrected, and how the fixes are verified.
* [docs/data_contract.md](docs/data_contract.md): input format for prepared data. **Start
  here if you prepare datasets.**
* [docs/design.md](docs/design.md): architecture, the patterns borrowed from Inspect AI and
  lmms-eval, and what is and isn't verified.
* [docs/project_brief.md](docs/project_brief.md): the original task brief.
* [logs/build_and_test.log](logs/build_and_test.log): the commands actually run and their
  output.

## Install

Python ≥ 3.10, on macOS or Linux (run locking uses `fcntl`).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'            # mock runs + tests
pip install -e '.[dev,gemini]'     # add the Gemini SDK for live runs
```

## Quick start (mock, offline)

```bash
python scripts/validate_data.py --benchmark benchmarks/mock_mcq_v1

python scripts/run_experiment.py run --config experiments/exp001_whole_mcq/config.yaml
python scripts/run_experiment.py run --config experiments/exp002_per_field_mcq/config.yaml

python scripts/evaluate.py --run runs/<run_id> \
    --ground-truth benchmarks/mock_mcq_v1/ground_truth.jsonl
```

Expected results on the synthetic report (worked out by hand; the mock answers are written
independently of the ground truth):

| experiment | calls | field accuracy | report completeness | all fields correct |
|---|---|---|---|---|
| exp001 whole report | 1 | 2/3 | 1/1 | 0/1 |
| exp002 per field | 3 | 1/3 (one response is not JSON → `invalid_output`) | 0/1 | 0/1 |

Cost is `unknown` because the mock reports no usage. It is never shown as 0.

`video-report …` is the same CLI, installed as a console script
(`video-report run|resume|status|evaluate|validate-data`).

## Experiments

Each experiment directory owns its config and prompts:

```
experiments/exp001_whole_mcq/
├── config.yaml        # data selection, method, prompt files, provider/model, generation, video, execution, pricing
└── prompts/{system,user}.txt
```

Relative paths in `config.yaml` resolve against the config file's own directory. Prompt
templates use `$questions`, `$answer_format`, `$field_ids`, `$num_fields` and `$context`; an
unknown placeholder is an error. Only the `input.context` keys listed in
`prompt.context_keys` reach the model.

To add an experiment, copy a directory, rename `name:`, and edit it. Never edit an
experiment's prompts to "fix" a run that already exists. Start a new run instead; the old one
keeps its frozen copy.

## Runs, parallel execution, resume

```bash
python scripts/run_experiment.py run --config <config.yaml> [--run-id ID] [--runs-dir runs] \
    [--video-root DIR] [--concurrency N] [--max-attempts N] [--timeout-s S]
python scripts/run_experiment.py status runs/<run_id>
python scripts/run_experiment.py resume runs/<run_id> [--config <config.yaml>] [--retry-failed] \
    [--allow-code-change]
```

* **Freezing.** When a run is created, the resolved config, prompt texts, selected reports,
  call plan, fully rendered requests, code revision (commit, dirty flag, diff hash) and
  package versions are written to `runs/<id>/snapshot/` and fingerprinted. Execution reads
  only this snapshot.
* **Parallel.** `execution.concurrency` caps in-flight calls within a run. Separate runs, even
  of the same experiment, are independent processes with separate directories. Run them in
  separate terminals or with `&`.
* **Checkpoints.** Each call has its own checkpoint file, rewritten atomically. Raw responses
  are saved before anything parses them, and every attempt is recorded.
* **Resume** continues pending calls and calls that were in flight during a crash.
  Successfully completed calls are never re-sent. `--retry-failed` also re-runs failed calls.
  Resume refuses to start if the frozen snapshot was modified, the package source code
  changed (a hash of all `video_report` `.py` files, including untracked ones; override with
  `--allow-code-change`), the provider/model/SDK changed, or, when `--config` is given, the
  current sources no longer match the frozen ones. Only `--concurrency`,
  `--max-attempts` and `--timeout-s` may differ.
* **One writer.** A second `run`/`resume` on the same run directory exits with code 5. A
  crashed writer never leaves a stale lock, because the OS releases `flock` on process exit.
* **Exactly-once is not guaranteed.** If the process dies after the provider has answered but
  before the response is saved, resume sends that call again. The first attempt is recorded
  as `interrupted` with an unknown outcome.

Exit codes: `0` completed, `2` invalid input / incompatible resume, `3` finished with failed
calls, `4` interrupted (resume to continue), `5` locked by another writer.

Ctrl-C and SIGTERM stop gracefully: in-flight attempts finish, queued calls are skipped, and
the run is marked `interrupted`. The writer lock is held until all workers are done. For
Gemini, one in-flight attempt can take up to upload + file processing + generation time.

## Offline evaluation

```bash
python scripts/evaluate.py --run runs/<run_id> --ground-truth <ground_truth.jsonl> \
    [--allow-markdown-fence] [--pricing pricing.json]
```

* No model calls are made. Results go to a new directory, `runs/<id>/evaluations/<eval_id>/`
  (`metrics.json`, `field_scores.jsonl`, `call_parses.jsonl`, `evaluation.json` with
  parser/scorer versions and the ground-truth sha256). Inference files are never modified.
* Field accuracy counts every selected field. Missing, invalid or not-executed answers count
  as incorrect.
* Also reported: report completeness, all-fields-correct rate, execution completion, provider
  failures by kind, and usage/cost with a status of `complete`, `lower_bound` or `unknown`.
* Partially executed runs are marked `provisional`, and the reason states the denominator.

`--pricing` takes a JSON file like
`{"input_per_million": 0.3, "output_per_million": 2.5, "source": "<pricing URL>, checked YYYY-MM-DD"}`.
Those numbers are only placeholders; take real prices from the provider's pricing page.

## Using teammate data

1. Prepare a benchmark directory that follows [docs/data_contract.md](docs/data_contract.md),
   kept outside Git.
2. Validate it: `python scripts/validate_data.py --benchmark /data/bench --check-videos --video-root /data/videos`
3. Point videos at your local copy without editing code. Either
   `export VIDEO_REPORT_VIDEO_ROOT=/data/videos` or pass `--video-root`. Precedence:
   CLI > env > config `data.video_root` > manifest `video_root` > benchmark directory.
4. Copy `experiments/exp001_whole_mcq`, set `data.benchmark: /data/bench`, and optionally
   restrict `data.report_ids` / `data.field_ids` for a small first run.

## Gemini (live)

```bash
pip install -e '.[gemini]'
export GEMINI_API_KEY=...            # only read from the environment
# edit experiments/exp003_whole_gemini_smoke/config.yaml:
#   provider.model: <a model id you checked in the current Gemini docs>
#   data.benchmark: <prepared data with real local videos>
python scripts/run_experiment.py run --config experiments/exp003_whole_gemini_smoke/config.yaml
```

What the adapter supports:

* Local video files, uploaded with the Files API.
* Optional clip offsets taken from `time_range`.
* Optional `video.fps`.
* `response_format: json`.

What it rejects at run creation: remote URIs and unknown MIME types.

SDK retries are disabled; the runner is the only retry layer. **Live calls cost money. Run
them only when authorized.** An opt-in live smoke test:

```bash
GEMINI_API_KEY=... LIVE_GEMINI_MODEL=<model id> LIVE_VIDEO_PATH=/path/clip.mp4 pytest -m live
```

## Development

```bash
pytest -q                      # offline suite; `live` tests are excluded by default
ruff check src tests scripts && ruff format --check src tests scripts
mypy                           # strict, on src/
```

CI (`.github/workflows/ci.yml`) runs lint, type checks and the offline tests on Python 3.10
and 3.12. A separate job runs without the Gemini SDK installed. No secrets are used.

Never committed: `.env`, credentials, real videos, downloaded datasets and `runs/*`.

## Status

What is verified, and how, is detailed in [docs/design.md §9](docs/design.md#9-reliability-properties-verified-vs-not)
and in the build log. In short:

* **Verified offline:**
  * mock end-to-end runs and evaluation
  * leakage guards
  * retries and failure classification
  * crash/resume, including a real process kill
  * single-writer locking across processes
  * snapshot immutability
  * concurrent CLI runs
  * Gemini request construction against a fake client
* **Not verified:**
  * any live Gemini call (upload, server-side clipping/fps, real usage, billing on timeouts)
  * Windows or network filesystems
  * large-scale performance
  * a fully green CI run on GitHub (partially observed; see the build log)

Passing tests do not make this production-ready.

Smoke-test runs automatically produce execution and evaluation logs; see
[logging workflow](docs/smoke_test_logging.md) and the
[five-video results and workflow](docs/smoke_test_results.md).
