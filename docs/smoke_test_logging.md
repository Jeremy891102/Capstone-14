# Standard smoke-test logging

Every completed, failed, or orderly interrupted execution session now writes
`runs/<run_id>/smoke_test.log` automatically. Resume refreshes the execution log.
A hard process kill may prevent log export; per-call checkpoint evidence remains on disk.
Preflight failures before a run is created do not have a run log.

Offline evaluation writes a separate, immutable
`runs/<run_id>/evaluations/<eval_id>/smoke_test.log` containing parsing, ground truth,
field scores and aggregate metrics. Evaluation does not modify the execution log.

Logs include frozen model-facing requests (prompts, selected fields, media path/range),
all recorded attempts and normalized provider responses, token usage, pricing,
per-call/aggregate estimated cost and observed provider-call latency. API credentials
in environment variables ending in API_KEY and recognized secret patterns are redacted.
Ground truth appears only in evaluation evidence, not in model-facing requests.

Recorded `latency_s` includes media preparation/cache lookup plus API round trip;
backend-only model inference time is not supplied. Missing usage is unknown, not zero.
Costs use configured list prices and are not account invoices. Evaluation pricing overrides
are explicitly shown in evaluation metadata/metrics and may differ from execution prices.

The P09 live validation additionally captures actual prepared MP4, wire request bodies,
raw HTTP JSON responses and separately instrumented media/API times under the local
outputs directory. Generic run logs reference input media rather than embedding base64.

Typical workflow:

1. Validate data and local videos.
2. Freeze one provider/model/input policy; verify pricing and paid-run scope.
3. Execute whole_report or per_field with an explicit run ID.
4. Inspect execution smoke_test.log, including failures.
5. Evaluate offline against the matching ground_truth.jsonl.
6. Inspect evaluation smoke_test.log for answer validity, scoring, usage and costs.
7. Compare methods by (report_id, field_id); retain failed/missing answers in denominators.

No API calls are made by log export or evaluation.
