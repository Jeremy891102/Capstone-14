# Capstone-14
Build and test an executable, extensible Python inference and evaluation codebase in this repository:

https://github.com/Jeremy891102/Capstone-14.git

Implement the code, run verification, document the results, and push the completed work to a dedicated branch. Do not stop at architecture recommendations.

## Project context

We study egocentric task video → structured reports with multiple fields.

Initially compare:
- One model call answering the entire report.
- Separate model calls answering individual fields.

Later, we may introduce grouped questions, type-based routing, OCR, tracking, and other tools. We do not train models.

Another teammate handles dataset preparation, potentially from HD-EPIC, CaptainCook4D, or other datasets. Do not invent dataset-specific conversion logic. Define and document a shared data contract, and use synthetic fixtures until real data arrives.

## Design goals

Keep the project understandable for a three-person team while providing the foundations needed for reliable experiments.

Consult current official documentation and relevant source code from Inspect AI and lmms-eval. Explain which design patterns you adopt. Do not claim our custom report schema is an industry standard.

Avoid unnecessary databases, plugin registries, distributed schedulers, or agent frameworks.

Use the existing repository root as the project root; do not create a redundant nested project directory.

Suggested structure:

src/video_report/
├── datasets/
│   └── base.py
├── benchmarks/
│   └── schema.py
├── providers/
│   ├── base.py
│   ├── mock.py
│   └── gemini.py
├── methods/
│   ├── whole_report.py
│   └── per_field.py
├── evaluation/
│   ├── parser.py
│   ├── scorers.py
│   └── aggregate.py
└── runner.py

experiments/
├── exp001_whole_mcq/
│   ├── config.yaml
│   └── prompts/
└── exp002_per_field_mcq/
    ├── config.yaml
    └── prompts/

benchmarks/mock_mcq_v1/
├── manifest.json
├── reports.jsonl
├── ground_truth.jsonl
└── mock_responses.jsonl

scripts/
├── run_experiment.py
└── evaluate.py

tests/
docs/data_contract.md
docs/design.md
logs/build_and_test.log
runs/
pyproject.toml
.env.example
.gitignore
README.md

Adjust this structure when justified, but explain changes. Do not create misleading placeholder converters.

## Module responsibilities

Use explicit, typed interfaces:

- Dataset reader: load prepared reports and validate model-facing data.
- Method: produce a call plan containing stable call IDs and selected field IDs.
- Prompt builder: render model input using the experiment’s prompt.
- Provider: perform one generation attempt and return raw output and available usage.
- Runner: execute calls, enforce concurrency and retry policies, persist progress, and resume.
- Evaluator: parse saved outputs, compare with targets, and aggregate scores.

Keep retry and scheduling logic in one place. Avoid duplicate retry layers in the SDK and runner.

## Data contract

Follow common evaluation naming conventions such as `id`, `input`, `choices`, `target`, and `metadata`, with a small custom report wrapper.

Support:
- Stable report and field IDs.
- One or more video references.
- Optional time ranges, with explicit units and coordinate semantics.
- Field question, answer type, and ordered choices.
- Source dataset and original annotation references.

Separate model-facing reports from ground truth and answer evidence. Methods, prompt builders, and providers must not receive targets or privileged annotation evidence.

Do not automatically send arbitrary metadata to the model. Whitelist any model-facing context.

Inference must work without ground truth. Evaluation requires valid targets for the selected fields.

Version one supports single-choice questions only. Explicitly reject unsupported types instead of silently mis-scoring them.

Provide one synthetic report with three fields and independent mock responses. The mock provider must not read ground truth to produce answers.

Document path resolution so teammates can use local videos without editing source code.

## Experiment isolation and reproducibility

Each experiment owns its configuration and prompts.

At run creation, freeze:
- Resolved configuration.
- Prompt contents.
- Selected report data and its fingerprint.
- Code revision and dirty-worktree status.
- Provider, model identifier, and generation settings.

Read frozen inputs during execution. Editing the original prompt or report file must not change a running experiment.

Use independent run directories, temporary files, and state. Never rely on mutable module-level experiment configuration.

Resume must verify compatibility and reject mismatched inputs or settings.

Do not introduce shared inference caches in version one.

## Execution and persistence

Implement:
- Configurable concurrency limits.
- Provider request timeouts and finite retry attempts.
- Classification of retryable versus permanent failures.
- Per-call checkpoints, including for per-field inference.
- Atomic persistence where appropriate.
- One active writer per run, with documented crash-safe lock behavior.
- Explicit run and call states.

Save raw responses before parsing. Preserve every recorded attempt and avoid duplicate final predictions during resume.

Distinguish:
- Valid but incorrect answers.
- Missing or invalid model output.
- Provider execution failures.
- Interrupted calls whose remote outcome is unknown.

Do not retry model format errors as ordinary network failures. Any output-repair strategy must be a separately configured method.

Do not promise exactly-once API execution: a crash after remote success but before local persistence may cause a duplicate call on resume.

API credentials must come from environment variables. Redact secrets and avoid writing signed URLs or credential-bearing exception contents to logs.

## Gemini integration

Use current official Google SDK documentation. Configure the model identifier rather than hardcoding a presumed latest model.

Mock execution must work without the Gemini SDK or API credentials.

Implement a minimal video inference path and document supported inputs. If clipping, FPS control, or another feature is not implemented, explicitly reject or document it rather than silently ignoring configuration.

Do not run paid live calls unless credentials and suitable video data are available and the execution is authorized. Clearly distinguish mocked SDK tests from actual live validation.

## Evaluation

Allow offline evaluation without new inference calls.

Store each evaluation separately with its scoring version/configuration and ground-truth fingerprint. Never overwrite original inference evidence.

For MCQ evaluation:
- Missing or invalid answers count as incorrect in field accuracy.
- Do not remove difficult or malformed answers from the denominator.
- Report execution completion and provider failures separately.
- Report field accuracy, report completeness, and all-fields-correct rate.
- For partially executed runs, label metrics provisional and state the denominator.
- Handle entirely failed runs without reporting misleading perfect or zero-cost results.

Track usage and cost per completed report, including additional attempts when known. Represent missing usage or pricing as unknown. Do not estimate unavailable usage as zero.

If a timed-out request might have been billed, state that observed usage may be a lower bound.

## Testing

Use pytest. Default tests must require neither network access nor API keys.

Unit tests:
- Duplicate IDs, invalid targets/choices, invalid time ranges, unsupported types.
- Prompt construction includes the intended fields and preserves choice mappings.
- Targets and annotation evidence never enter model requests.
- Valid JSON, missing/extra fields, invalid options, duplicate keys, empty responses.
- Scoring against independently hand-calculated expected values.
- Whole-report and per-field call planning.
- Out-of-order completions matched by IDs.
- Retry limits and permanent-error handling.
- Immutable run snapshots and resume compatibility.

Integration tests:
- Complete mock inference followed by offline scoring.
- Concurrent experiments with distinct prompts and outputs.
- Two CLI processes running without cross-run contamination.
- A second writer rejected for the same run.
- Mid-run source-prompt edits do not affect frozen requests.
- Injected interruption followed by resume.
- Persisted successful calls are not repeated.
- Attempt records and final predictions remain consistent.
- All-failure and partial-run accounting.

Use prompt snapshots where useful. Test failure behavior, not just happy paths.

Inject clocks, sleeps, or scripted provider failures when needed so tests are deterministic and fast.

Mark live API tests separately and exclude them by default. Do not assert that a real model always answers correctly as a software-test requirement.

Add a lightweight CI workflow for linting, type checking, and offline tests. It must not require API secrets.

## Build and test log

Produce and commit `logs/build_and_test.log`.

Record observable actions and results:
- UTC timestamp and stage.
- Files created or modified and their purpose.
- Exact verification commands.
- Actual output, exit codes, and test summaries.
- Failures, fixes, and rerun results.
- Validated capabilities and remaining limitations.

Preserve failed test output. Do not fabricate successful runs, coverage, or checks. Do not include internal reasoning or secrets.

## Repository workflow

1. Inspect the repository, existing files, and applicable AGENTS.md instructions.
2. Create a dedicated branch, such as `feat/video-report-pipeline`.
3. Preserve unrelated work and repository history.
4. Implement incrementally:
   schema → parser/scorer → mock flow → isolation/resume → Gemini interface.
5. Run tests and relevant checks, then perform a final code review.
6. Commit code, documentation, synthetic fixtures, CI configuration, and the build/test log.
7. Exclude credentials, real videos, downloaded datasets, and runtime outputs from Git.
8. Push the branch to the supplied repository; do not force push or merge into the default branch.

If authentication or network access blocks pushing, finish all local work and report the exact blocker and remaining command. Never claim a push succeeded without verification.

## Deliverables

- Executable codebase.
- Data contract and synthetic examples.
- README covering installation, mock runs, parallel execution, resume, offline evaluation, and teammate-data integration.
- Design notes explaining adopted patterns and limitations.
- Build/test log.
- Commit hash and pushed branch link, if successful.
- Concise final report of actual test results and unvalidated features.

Passing tests does not alone establish production readiness. State precisely which reliability properties were verified.

The practical goal is that once our teammate supplies prepared data, we can adapt the reader or map the fields, validate the data, and begin a real smoke test without restructuring the entire project.
