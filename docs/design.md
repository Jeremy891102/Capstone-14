# Design notes

These notes cover what the code does, why, which ideas come from established eval frameworks,
and what has **not** been verified.

## 1. Scope

The task is to answer structured reports about egocentric task videos and to compare
*whole-report* calls (one call answers every field) against *per-field* calls (one call per
field). We call models and do not train them.

v1 supports single-choice fields only, with two providers: `mock` (offline) and `gemini`.

Deliberately out of scope: databases, plugin registries, distributed schedulers, agent
frameworks, shared inference caches, and dataset-specific converters.

## 2. Module map

```
src/video_report/
├── benchmarks/schema.py   data contract v1 (pydantic models, strict, extra="forbid")
├── datasets/base.py       JSONL readers, target validation, video path resolution
├── config.py              experiment config model + YAML loading
├── methods/               call planners: whole_report.py, per_field.py (+ check_plan)
├── prompting.py           template rendering (string.Template), context whitelist
├── providers/
│   ├── base.py            ModelRequest / ProviderResponse / Usage, error classes, redaction
│   ├── mock.py            canned-answer provider + fault injection
│   └── gemini.py          google-genai adapter (Files API + generate_content)
├── snapshot.py            build frozen run inputs from sources; code revision capture
├── run_store.py           run directory layout, flock writer lock, checkpoints
├── runner.py              THE retry/concurrency/checkpoint loop
├── pipeline.py            create / resume (compatibility checks) / status
├── evaluation/
│   ├── parser.py          raw text -> per-field raw answers (pure)
│   ├── scorers.py         one field -> outcome (pure)
│   ├── aggregate.py       metrics, execution accounting, usage/cost (pure)
│   └── evaluate.py        offline evaluation of a run directory -> evaluations/<id>/
└── cli.py                 argparse CLI; scripts/*.py are thin wrappers
```

How this differs from the suggested layout, and why:

* `prompting.py`, `config.py`, `snapshot.py`, `run_store.py` and `pipeline.py` are separate
  modules instead of being folded into `runner.py`. That keeps `runner.py` limited to
  execution policy, so retries live in exactly one small file. It also lets snapshot freezing
  and the run layout be tested on their own.
* `evaluation/evaluate.py` holds the offline-evaluation entry point. `scripts/evaluate.py`
  only calls it.
* `scripts/validate_data.py` was added so a teammate can validate prepared data without
  running anything.
* `experiments/exp003_whole_gemini_smoke/` is a template for live runs. Its model is set to
  the placeholder `SET-ME`, which is rejected until someone fills in a real model id.
* There is no `datasets/<name>.py` converter. That would be dataset-specific logic, and it
  belongs to data preparation.

## 3. Data flow and the target firewall

```
reports.jsonl ──► select reports/fields ──► method.plan ──► CallSpec(call_id, report_id, field_ids)
                                                   │
experiment prompts + context whitelist ──► render ─┴─► ModelRequest (frozen to snapshot/requests.jsonl)
                                                            │
                                                  provider.generate (one attempt)
                                                            │
                                        responses/<call>/attempt-N.json (raw, before parsing)
                                                            │
ground_truth.jsonl ──────────────► evaluation: parse ► score ► aggregate ► evaluations/<id>/
```

Targets and evidence exist only in `ground_truth.jsonl`. They are read by `evaluation/` and
by `validate-data`, and by nothing else. The structure of the code enforces this:

* `Report` forbids unknown keys, so `target`/`answer`/`evidence` in `reports.jsonl` is a
  validation error.
* Methods receive `ReportSelection` (ids only). The prompt builder receives `FieldSpec`s
  (question + choices) and the context dict filtered by `prompt.context_keys`. Providers
  receive `ModelRequest`, which has no target field.
* `source`, `metadata` and non-whitelisted `context` keys are never rendered. Tests assert
  that sentinel strings placed in evidence, metadata, source and non-whitelisted context never
  appear in any serialized request (`tests/test_prompting_and_planning.py`).
* The mock provider reads only its frozen `mock_responses.jsonl`. The committed mock answers
  deliberately disagree with the ground truth on one field.

## 4. Runs: freezing, states, persistence

### Freezing (run creation)

`snapshot.build_snapshot` resolves everything once: the config, prompt **contents**, the
selected reports (restricted to selected fields), the call plan, fully rendered requests
(including resolved video paths), and the mock responses. These are written to
`runs/<id>/snapshot/` together with:

* `fingerprints.json`: sha256 of canonical JSON per component (`config` = the
  inference-relevant view, which excludes `execution`, `description` and `pricing`), plus
  `prompts`, `reports`, `call_plan`, `requests`, `mock_responses` and `overall`.
* `environment.json`: `source_sha256`, a hash over every `.py` file of the installed
  `video_report` package (path + content), which works with or without git and includes
  untracked files. It also records the git commit, a dirty flag, and a hash of uncommitted
  changes: `git diff HEAD` plus the contents of untracked, non-ignored files, hashes only.
  Finally the Python version, platform and package versions.

Execution reads **only** `snapshot/requests.jsonl` and the frozen config. Editing a source
prompt, report or config during or after a run has no effect on that run. This is tested both
in-process and with a separate CLI process editing the prompt mid-run.

Run ids are `<experiment>-<UTC timestamp>-<random hex>`, or `--run-id`. The run directory is
created with an exclusive `mkdir`, so two processes can never share one. Nothing is
module-level mutable state. The CLI integration test runs two experiments concurrently as
separate processes and checks that requests and predictions do not cross over.

### States

The **run state** is stored in `run.json`:
`created → running → completed | completed_with_failures | interrupted`.
A process that crashes leaves `running` behind. `status` reports this as
`running (stale: no active writer …)` by probing the lock.

The **call status** is stored in `calls/<key>.json`:
`pending → in_progress → succeeded | failed` (a retryable error goes back to `pending`
between attempts).

Every attempt is recorded and kept, with one of these outcomes:

| outcome | meaning |
|---|---|
| `in_progress` | written *before* the provider call |
| `success` | raw response persisted |
| `retryable_error` | transient failure. For 429/5xx and connect failures the request was not processed. For a connection that broke after sending (read error, protocol error), the error carries `outcome_unknown: true` and counts as possibly billed |
| `timeout` | no response in time; **remote outcome unknown, may have been billed** |
| `permanent_error` | provider rejected the request (4xx, missing file) or an unexpected adapter exception occurred (the latter is `outcome_unknown: true`); not retried |
| `interrupted` | the process died mid-attempt (found on resume); **remote outcome unknown** |

### Persistence protocol

1. Append an `in_progress` attempt and write the call file (atomically: temp file, fsync,
   `os.replace`, then fsync of the directory).
2. Call the provider.
3. On success, write `responses/<key>/attempt-NNNN.json` atomically **first**, then mark the
   attempt `success` and the call `succeeded`.

On resume, a call left `in_progress` is either **recovered** (its response file exists, so no
new request is made) or marked `interrupted` and re-queued. `predictions.jsonl` is rebuilt
from the call files at the end of each session, so it has exactly one row per succeeded call
and resuming cannot duplicate predictions.

**At-least-once, not exactly-once.** If the process dies after the provider has processed the
request but before step 3 writes the response, the call is re-sent on resume and may be billed
twice. The `interrupted` attempt record makes this visible.

### Writer lock

`fcntl.flock(LOCK_EX | LOCK_NB)` is taken on `run.lock` for the whole session. The
acquisition is retried for about 1 s, because `status`/`evaluate` probe the lock with a
momentary shared lock (and never create the file); that probe must not be mistaken for a
writer. After Ctrl-C/SIGTERM the runner keeps the lock until every worker thread has
finished (further interrupts are ignored while draining). This prevents a concurrent `resume`
from racing with late checkpoint writes. The kernel
releases a `flock` when the process exits for any reason, including SIGKILL and `os._exit`.
A crash therefore never leaves a stale lock, and no PID-liveness heuristics are needed. A
second writer gets exit code 5. This is tested in-process and across processes.

Limitations: POSIX only (no Windows), and `flock` is unreliable on some network filesystems
(NFS). Keep run directories on local disk.

### Resume compatibility

`resume` always executes the frozen snapshot. Before doing so it:

1. recomputes the fingerprints of the frozen files and rejects the run if they were modified;
2. compares the current package `source_sha256` with the frozen one. A mismatch, or an
   unknown frozen hash, is rejected unless `--allow-code-change` is passed (the decision is
   recorded in the session notes). A different git commit with identical package source is
   only a note;
3. compares `provider.describe()` (provider, model, SDK version) with the value recorded at
   the first session, and rejects any mismatch;
4. with `--config`, rebuilds a snapshot from the current sources and rejects the resume if any
   component fingerprint differs.

Execution settings (`concurrency`, `max_attempts`, `timeout_s`) may be overridden on resume.
They do not change what is asked, and overrides are recorded per session.

## 5. Execution policy (single retry layer)

* **Concurrency:** a `ThreadPoolExecutor(max_workers=execution.concurrency)`. Provider calls
  are I/O-bound, and threads keep the code simple and synchronous. Each call file is written
  only by the worker that owns the call; `run.json` is written only by the main thread.
* **Retries:** handled only in `runner.Runner._execute_call`. The budget is `max_attempts`
  per call per session, with exponential backoff
  `min(max, initial·2^(k−1)) · (1 + jitter·U[0,1))`. Sleep and RNG are injectable, so tests
  run with zero real waiting.
* **Classification:** done in each provider adapter, which maps exceptions to
  `RetryableProviderError`, `ProviderTimeoutError` (retryable, `outcome_unknown`) or
  `PermanentProviderError`. Only connection failures (`ConnectError`, `ConnectTimeout`) and
  HTTP error responses count as "request not processed". Read and protocol errors after
  sending, timeouts, and unexpected exceptions are `outcome_unknown`. An unknown exception is
  recorded as a permanent `unexpected_error` and is **not** retried.
* **SDK retries disabled:** the Gemini client and each request use
  `HttpRetryOptions(attempts=1)`. One caveat: in google-genai 2.28.0 the resumable-upload
  chunk loop has its own small fixed retry for unfinalized chunks, and that cannot be
  configured. It affects uploads only.
* **Format errors are not retried.** A response that is not valid JSON is still a successful
  provider call. It is saved and later scored `invalid_output`. Any output-repair strategy
  (such as re-asking) would have to be a separately configured method; none exists in v1.
* **Timeouts:** `execution.timeout_s` is passed to the provider and applied to each HTTP
  request (Gemini: upload and `generate_content`, in ms). The runner cannot kill a thread, so
  after Ctrl-C/SIGTERM it waits for in-flight attempts, cancels queued calls, and marks the
  run `interrupted`. For Gemini, one attempt can take up to the upload timeout, plus
  `file_active_timeout_s` (default 300 s) of polling, plus the generation timeout. It is
  **not** bounded by a single `timeout_s`.

## 6. Evaluation

Evaluation works from the run directory alone (the frozen reports and saved raw responses)
plus a ground-truth file, and makes no provider calls. Each evaluation is written to a new
directory, `evaluations/<eval_id>/`. Inference files are never modified; a test compares
file hashes before and after.

| file | content |
|---|---|
| `evaluation.json` | parser/scorer versions, scoring options, pricing used, GT file sha256 + selected-target fingerprint, run fingerprint |
| `field_scores.jsonl` | one row per selected field: outcome, predicted letter/text, target |
| `call_parses.jsonl` | per-call parse status, missing / extra keys |
| `metrics.json` | aggregate metrics (below) |

Field outcomes are `correct`, `incorrect`, `invalid_answer` (not one of the field's
letters), `missing_answer`, `invalid_output` (empty, invalid JSON, duplicate keys, not an
object) and `not_executed`. **Every selected field stays in the denominator.**

Metrics:

* `field_accuracy` = correct / all selected fields (numerator and denominator are always
  reported).
* `valid_answer_rate` = (correct + incorrect) / all fields.
* `report_completeness_rate` = reports where every field has a valid answer / all reports.
* `all_fields_correct_rate` = reports with every field correct / all reports.
* `execution`: `execution_completion` (succeeded calls / planned calls), calls by status,
  failed calls by reason, and attempt outcomes and error kinds. These are reported separately
  from accuracy.
* `provisional: true` when any call has not succeeded or a writer is active. The reasons
  state the denominator.
* When no call succeeded, a warning says that accuracy 0 reflects execution failure, and cost
  is not reported as 0.

### Usage and cost

* Usage is taken per attempt from the provider (Gemini `usage_metadata`). A missing value is
  `None` (unknown) and is never counted as 0.
* `cost_status` is `complete` only if every billable attempt has known usage and pricing is
  configured. It is `lower_bound` if any attempt timed out or was interrupted (the request
  may have been processed and billed), or succeeded without usage. It is `unknown` if pricing
  is missing or no usage is known at all.
* Error attempts where the provider returned an HTTP error, or the connection could not be
  opened, are **assumed unbilled**. This is an assumption, and the attempt counts are reported
  so it can be audited. Errors marked `outcome_unknown` make cost a lower bound.
* `cost_per_completed_report` covers reports whose calls all succeeded, and includes every
  attempt of those calls (retries too).
* Pricing comes from the experiment config (`pricing:`, with a required `source` string such
  as the URL and date checked) or `--pricing file.json`. No prices are hardcoded.
  `prompt_token_count` is treated as including cached tokens. If cached tokens are present but
  no cached price is set, cost is `unknown`.

## 7. Gemini adapter (minimal video path)

The adapter's inputs and outputs:

* **Input:** local video files only. Each is uploaded with `client.files.upload`, then
  `files.get` is polled until the file is `ACTIVE`. The upload is reused within one provider
  instance (keyed by path + size + mtime, in memory) and deleted on `close()`.
* **Not supported:** remote URIs (`gs://`, `https://`, YouTube) and unknown or unsupported
  MIME types. These are **rejected at run creation**, before any call.
* **Clipping and fps:** `time_range` is sent as `VideoMetadata(start_offset="12.0s",
  end_offset="48.5s")`, and `video.fps` as `VideoMetadata.fps` (the SDK documents the range
  as (0, 24]). The adapter sends these, but no live call in this repository has checked that
  the server honours them.
* **Generation settings:** `temperature`, `top_p`, `max_output_tokens` and `seed` are passed
  through. `response_format: json` sets `response_mime_type="application/json"`. There is no
  response schema or enum-constrained decoding in v1.
* **Output text:** the concatenated non-thought text parts of the first candidate. The adapter
  also records the finish reason, model version, response id and block reason, if any.
* **Credentials:** read from the env var named by `provider.gemini.api_key_env` (default
  `GEMINI_API_KEY`). Error messages pass through `redact()`, which removes API keys, `key=` /
  `token=` / `Authorization` values and URL query strings (signed URLs), plus the literal key.
* **Model id:** always explicit in the config. Nothing defaults to a "latest" model.

The adapter was written against the installed `google-genai` 2.28.0 source: `HttpOptions.timeout`
is in milliseconds, `retry_options=None` means no retry, `VideoMetadata` has
`start_offset`/`end_offset`/`fps`, and `UsageMetadata` has `prompt_token_count`,
`candidates_token_count`, `thoughts_token_count`, `cached_content_token_count` and
`total_token_count`.

**Validation status:** unit-tested with a fake client built from the real SDK `types`.
**No live API call was made while building this repository.** No credentials or real videos
were available, and paid calls were not authorized. `tests/test_live_gemini.py`
(`pytest -m live`) is the opt-in smoke test.

## 8. Patterns adopted from Inspect AI and lmms-eval

I checked these against the source of **inspect-ai 0.3.276** and **lmms-eval 0.7.3**
(downloaded wheels) plus the Inspect docs. Paths below are relative to each package.

| pattern | where it comes from | how we use it |
|---|---|---|
| `id` / `input` / `choices` / `target` (letter) / `metadata` naming | Inspect `Sample` (`inspect_ai/dataset/_dataset.py`); multiple-choice targets are letters, mapped by position (`scorer/_choice.py`) | our report and ground-truth keys; letter targets with optional `target_text` cross-check |
| Model input built without the target | lmms-eval `construct_requests` passes `(ctx, gen_kwargs, doc_to_visual, …)`, not `doc_to_target` (`api/task.py`) | `ModelRequest` has no target field; GT lives in a separate file |
| Solver vs scorer separation; scorer is a pure function | Inspect `Solver` / `Scorer` (`solver/_solver.py`, `scorer/_scorer.py`) | provider/runner never parse; `parser.py` / `scorers.py` are pure |
| Score with a machine-readable reason | Inspect `Score.reason` (`invalid_response_format`, `no_response`, …) | explicit field outcomes (`invalid_output`, `missing_answer`, …) |
| Raw outputs logged, scoring re-runnable offline | Inspect `--no-score` + `inspect score` (`_cli/score.py`, writes a separate scored log unless `--overwrite`); lmms-eval `--log_samples` | raw responses saved before parsing; each evaluation in its own directory, never overwritten |
| Run status values | Inspect `EvalStatus = started/success/cancelled/error` (`log/_log.py`) | our run states (`running/completed/completed_with_failures/interrupted`) |
| Reproducibility record | Inspect `EvalSpec.revision` (git commit + dirty) and packages; lmms-eval results `git_hash`, config | `snapshot/environment.json` + fingerprints |
| Resume reuses only successful, id-matched samples | Inspect `eval_retry` reuses samples keyed by `(id, epoch)` with no error (`_eval/task/run.py`) | only `succeeded` calls are skipped; matched by stable `call_id` |
| Central retry with provider-specific classification | Inspect `model/_retry.py` + per-provider `should_retry`; Google provider treats 408/429/5xx as retryable, 4xx not (`model/_providers/google.py`) | same status classes; one retry loop in `runner.py` |
| Usage with optional fields that are not zero | Inspect `ModelUsage` (`reasoning_tokens`, cache fields optional) | `Usage` with `None` = unknown |
| Fingerprint config/prompt so changes invalidate reuse | lmms-eval response cache key includes a task fingerprint (`caching/response_cache.py`) | snapshot fingerprints gate resume (we do **not** cache responses) |

Deliberately **not** adopted:

* **Two retry layers.** Inspect leaves SDK default retries on for some providers (OpenAI,
  Anthropic). We disable SDK retries so attempts are counted and billed exactly once in our
  records.
* **Unlimited retries by default** (Inspect `max_retries=None`). We default to 3 attempts and
  allow at most 10.
* **Error strings as model answers.** In lmms-eval, API models return
  `"[LMMS_EVAL_REQUEST_FAILED after N retries] …"` as the response text (e.g.
  `models/simple/gemini.py`), which could then be cached or scored like an answer. We record
  structured errors, and a failed call's fields are `not_executed`.
* **Response caches** (lmms-eval `--use_cache` SQLite + JSONL). These were excluded by
  requirement in v1.
* Registries, `!function` YAML indirection, adaptive concurrency, sandboxes, epochs/reducers,
  and hub uploads. All of this is more than a three-person project needs.

## 9. Reliability properties: verified vs not

Verified by the offline test suite (`pytest`: 167 passed on Python 3.13.9 and on 3.10.22; on 3.10.22 without the SDK, Gemini adapter tests skip and the rest pass,
see `logs/build_and_test.log`):

* Schema rejects duplicate IDs, invalid targets/choices/time ranges, unsupported types and
  target-like keys in reports (`test_schema.py`).
* Prompts contain exactly the requested fields with order-preserving letters; context
  whitelist; exact prompt snapshots (`test_prompting_and_planning.py`).
* Targets, evidence, metadata, source and non-whitelisted context never appear in requests
  (same file).
* Parser and scorer handle valid JSON, missing and extra fields, invalid options, duplicate
  keys, empty responses and fences (`test_parser.py`).
* Aggregates match hand-calculated values; all-failed and partial runs; unknown and
  lower-bound cost (`test_aggregate.py`, `test_evaluation.py`).
* Out-of-order completion is matched by id; retry limits; permanent errors and format errors
  are not retried; unexpected adapter errors are not retried and are redacted
  (`test_runner.py`).
* A crash (in-process `BaseException`, and a real `os._exit` in a CLI subprocess) followed by
  resume: successful calls are not repeated and their checkpoints are byte-identical; the
  crashed attempt is recorded as `interrupted`; predictions have no duplicates; recovery from
  a persisted response makes no new request.
* A second writer is rejected (in-process and cross-process); the lock is released after a
  crash.
* Mid-run source prompt edits do not affect frozen requests; resume rejects changed sources,
  tampered snapshots, a changed code revision and a changed provider description.
* Two concurrent CLI runs show no cross-contamination.
* Mock runs work with the `google` package import blocked.
* Field-subset runs evaluate against a full ground-truth file. Package source changes,
  including untracked files, are detected on resume. A `status` probe does not block a
  starting writer. The drain loop survives repeated interrupts (unit-level, with a simulated
  pool).
* The Gemini adapter builds requests correctly, maps errors and redacts secrets — against a
  **fake** client.

**Not** verified:

* Any live Gemini call: Files API upload behaviour, whether server-side clipping and fps are
  honoured, real usage numbers, real error payloads, and billing on timeouts.
* Behaviour on Windows or network filesystems (locking).
* Large-scale performance: thousands of calls, or `read_all_calls` overhead (it reads every
  call file).
* Durability under power loss beyond what `fsync` + `os.replace` provide on the local
  filesystem.
* The CI workflow on GitHub-hosted runners. It was written but not observed running at the
  time of this commit.

## 10. Extension points (later work)

* **Grouped questions / type-based routing:** add a planner in `methods/` that groups
  `field_ids` per call (`check_plan` already enforces exactly-once coverage), plus a new entry
  in `METHODS`.
* **New answer types:** extend `SUPPORTED_ANSWER_TYPES` and add a parser/scorer pair with new
  version strings. Old evaluations keep their recorded versions.
* **Tools (OCR, tracking):** best modelled as a separate preprocessing step whose outputs are
  frozen into the snapshot as additional model-facing context, so they are fingerprinted like
  everything else.
* **Output repair:** a separate method/config that issues a second, explicitly recorded call.
  It must never be a hidden retry.
