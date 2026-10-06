# OpenRouter response and token-accounting fixes

## Why modify

### Reasoning was charged twice

OpenRouter's `usage.completion_tokens` includes reasoning tokens. The adapter previously
stored that total as `Usage.output_tokens` and also stored `reasoning_tokens`. Our evaluator
adds `output_tokens + reasoning_tokens`, so completion=30 and reasoning=20 were charged as
50 output tokens instead of 30. This distorted method-cost comparisons even when inference
answers were correct. It did not cause an extra API charge; it overstated our local estimate.

### HTTP 200 did not guarantee a valid API envelope

A JSON decoding failure was converted to `{}`, then returned as a successful response with
no text. JSON arrays or malformed `choices`/`message` values could instead raise unclassified
exceptions. Those are provider-response failures, not model-answer format errors. Treating
an invalid envelope as success falsely increased execution completion and hid the cause.

## How to fix

* Normalize known completion/reasoning counts in the OpenRouter adapter:
  `output_tokens = completion_tokens - reasoning_tokens`. For 30 total completion and
  20 reasoning, store output=10 and reasoning=20. The existing evaluator then charges 30.
* Keep absent token fields unknown. When a reasoning breakdown is absent, preserve the
  reported completion total and reasoning=None; evaluation uses the reported total once.
  Reject negative/non-integer counts and reasoning greater than completion.
* Keep HTTP 4xx/5xx error classification independent of JSON decoding. Embedded OpenRouter
  error objects still receive their provider-error classification.
* Before accepting HTTP 200, require an object with nonempty object choices, an object
  message, and string/null content (or an explicit refusal). Validate optional usage and
  identifiers so malformed fields cannot cause an unclassified parsing crash.
* Malformed success envelopes raise `PermanentProviderError(kind="invalid_response",
  status_code=200, outcome_unknown=True)`. Runner records a failed attempt and does not
  automatically resend: the remote request may already have run and been billed. Here
  "permanent" means no automatic retry in this run, not that the service can never recover.
* Preserve empty/null/non-JSON **model answer text** in an otherwise valid envelope for the
  existing offline scorer. The adapter does not parse or repair report answers.
* Persist fixed diagnostic messages rather than arbitrary gateway bodies, which may echo
  input data or credentials.

## Verification

Offline tests use `httpx.MockTransport`, synthetic video and real local ffmpeg. No model API
requests, real credentials, or API credits are used. Regression tests cover the hand-calculated
cost, missing/all-reasoning usage, malformed envelopes, invalid usage, null/empty/non-JSON
model answers, embedded errors and HTTP failures. A pipeline-level test verifies that a
malformed HTTP 200 produces one failed checkpoint even with max_attempts=3, zero execution
completion, a not_executed field and possibly-billed accounting.

Commands and actual results are recorded in `logs/openrouter_response_fixes.log`.
Old run checkpoints and evaluations are not rewritten by this change; previously recorded
inflated usage needs separate correction before historical cost comparisons.

Source: https://openrouter.ai/docs/guides/best-practices/reasoning-tokens
Checked 2026-10-06: completion tokens include reasoning; the documented visible-output count
is completion minus reasoning.
