# Three API sources and the localization pilot

This branch adds direct OpenAI Responses and Dennis's Vertex Express Gemini endpoint.
Existing OpenRouter and Gemini Developer adapters remain available. All verification is offline;
real account access, key limits, model availability, billing and real-video behavior are unverified.
No key or paid request is required for tests. No live requests were made.

## Fill keys locally

The ignored `.env` file has been created only if it did not exist. Fill these fields locally:

```text
OPENROUTER_API_KEY=
OPENAI_API_KEY=
GEMINI_API_KEY=
VIDEO_REPORT_VIDEO_ROOT=/data/HD-EPIC/Videos
```

Never commit `.env` or paste keys into chat. Its permissions are 0600 when created.
The CLI still reads environment variables, not `.env` automatically. In the execution shell:

```bash
set -a
source .env
set +a
python scripts/check_credentials.py
# Alternatively check file presence without sourcing or making any API request:
python scripts/check_credentials.py --env-file .env
```

The checker prints presence only; it cannot validate account access or prove the $100/$500 caps.
There is no automatic spending-cap enforcement in the runner. Keep max_attempts=1 for the initial
smoke test. Do not run model calls until the user authorizes paid testing.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,openrouter,openai,vertex]'
# Add the gemini extra only to use the original Developer SDK adapter.
```

The two new adapters use httpx directly, with no hidden HTTP retries or redirects. ffmpeg is
bundled by imageio-ffmpeg. Local preparation has a separate timeout; runner owns API retries.

## Configuration

Templates use one report, eight questions, concurrency=1, max_attempts=1 and pricing=null:

* `experiments/exp005_loc_openrouter/config.yaml`: provider=openrouter, video_url inline clip.
* `experiments/exp005_loc_openai/config.yaml`: provider=openai, fixed GPT-4.1 snapshot
  `gpt-4.1-2025-04-14`, timestamped JPEG frames through https://api.openai.com/v1/responses.
  Choose an image-capable Responses model if changing it. Model access is not verified.
* `experiments/exp005_loc_vertex/config.yaml`: provider=gemini, gemini.backend=vertex_express,
  Dennis's exact model ID `gemini-3.5-flash-lite`, fixed endpoint
  https://aiplatform.googleapis.com/v1/publishers/google/models/gemini-3.5-flash-lite:generateContent,
  with x-goog-api-key authentication. No project/location or OAuth token is required by this
  Express endpoint. This is a different service from the Developer Files API.

Original provider=gemini configs default to backend=developer and keep their existing behavior.
The same GEMINI_API_KEY environment name is used, but the key must belong to the configured
backend. Do not run a Developer config with Dennis's Vertex Express key.

All report video references are local files plus time_range. Original videos remain reusable;
short clips/frames are process-local temporary assets and cleaned on normal close. Per-field
calls reuse local preprocessing but still send media with each model request. Source video bytes
are not frozen in existing snapshots; do not replace originals between sessions.

## Media and comparability

The new Vertex adapter clips the specified interval, downscales to 480 pixels high, resamples
at 2 FPS and removes audio. This matches the existing OpenRouter visual-only input policy.
OpenAI extracts JPEG frames at the same configured FPS/height, adds explicit clip-relative
sample-time labels, and sends no audio. Labels describe the fixed sampling timeline, not action
annotation evidence. Request.fps overrides the provider default explicitly.

The first frame is sampled at clip-relative 0; question timestamps in the prepared HD-EPIC
benchmarks are already relative to this clip. A frame limit or payload limit causes failure,
never silent reduction in FPS. Default OpenAI guard: at most 1500 frames total, 40 MB payload.
Default Vertex application payload guard: 20 MB. Account/model limits can be stricter.
A local ffmpeg failure occurs before any HTTP model request. A malformed success envelope is
classified invalid_response with unknown remote outcome, rather than a false successful answer.
Model-output format errors still belong to offline scoring.

OpenAI is a frame-based baseline, Gemini/OpenRouter are native-video baselines. Their inputs
are NOT identical merely because FPS/height match: upstream video sampling and temporal/audio
processing differ. Analyze report-size effects within a fixed provider and input policy first;
label cross-provider comparisons with the modality difference. Subsecond actions may be missed
at 2 FPS; preview media and choose a fixed policy before any paid experiment.

Usage normalization keeps reasoning separate without charging it twice. OpenAI output_tokens
includes reasoning; subtract the known breakdown before handing usage to the existing evaluator.
Vertex candidatesTokenCount and thoughtsTokenCount are recorded separately. Missing counts stay
unknown. Pricing is deliberately unset until the exact account/model prices are checked.

## Single-type nested design

`data/hd_epic_loc_nested_v1` contains three videos from different participants (P09, P02, P04),
96 unique localization questions, 8 ⊂ 16 ⊂ 32 sets, and three orders per size.
Whole-report inference uses the nine loc_nXX_orderY folders. The loc_per_field folder contains
all 32 questions per video for the one-question-at-a-time baseline. Do NOT repeat the per-field
baseline separately for each size: reuse those same per-question results for paired comparisons.

The three orders are seeded counterbalanced permutations: shuffle the largest set, rotate it
three ways, then filter smaller subsets. Every retained question has three distinct positions,
spanning at least two position thirds. These are not three independent random-sampling replicates
or a test of model stochastic variance. Video intervals, question text, choices and targets do
not change across sizes/orders. Temporal coverage uses annotation evidence midpoints and is
approximately balanced where possible; largest-set thirds are shown in summary.json.

Named repeated actions (with simple put/leave normalization) and overlapping target evidence
are excluded. This is not a guarantee of semantic independence. All sets are marked draft:
review question_review.md against original video, check timestamp boundaries and semantic hints,
and approve replacements across every size/order together. Sampling uses target timestamps only
for offline stratification; answers/evidence never enter prompts. Original benchmark coverage
is restricted further, so results do not represent the full HD-EPIC VQA leaderboard.

Rebuild without overwriting prior review work:

```bash
PYTHONPATH=src python scripts/build_localization_pilot.py \
  --source data/hd_epic_q40_v1 --out /path/to/new-pilot --seed 20261007
```

Complete intended batch per API source: 3 videos × 3 sizes × 3 orders = 27 whole-report calls,
plus 3 × 32 = 96 per-field calls, or 123 calls before retries. This is a plan, not an executed
experiment. Across three sources it would be 369 calls; approve an estimated budget first.

## After key entry and video review

Validate data and files, without calling APIs:

```bash
python scripts/validate_data.py \
  --benchmark data/hd_epic_loc_nested_v1/loc_n08_order1 \
  --check-videos --video-root /data/HD-EPIC/Videos
```

When paid testing is explicitly authorized, the run command is:

```bash
python scripts/run_experiment.py run --config experiments/exp005_loc_vertex/config.yaml
# Swap config to exp005_loc_openai or exp005_loc_openrouter for that source.
```

Evaluation is offline:

```bash
python scripts/evaluate.py --run runs/<run_id> \
  --ground-truth data/hd_epic_loc_nested_v1/loc_n08_order1/ground_truth.jsonl
```

To expand sizes/orders, copy an experiment config and change data.benchmark; keep model,
video input policy and generation settings fixed. For per-field use loc_per_field and
method.name=per_field. Do not concatenate benchmark files: stable report IDs repeat between
variants and each run must have its own directory.

## Sources checked 2026-10-07

* https://developers.openai.com/api/docs/guides/images-vision
* https://developers.openai.com/api/docs/guides/reasoning
* https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=responses
* https://developers.openai.com/api/docs/models/gpt-4.1
* https://docs.cloud.google.com/gemini-enterprise-agent-platform/reference/express-mode/rest/v1/publishers.models/generateContent

Key caps ($100 OpenRouter, $500 OpenAI, $500 Gemini) come from the mentor's message, not
verified account queries. No value is assumed to be available credit merely because a key exists.
