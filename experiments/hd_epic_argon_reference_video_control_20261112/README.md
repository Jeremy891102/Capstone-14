# HD-EPIC Argon: reference context × video control

**Question:** Does adding seven reference questions help a target answer more when video is available than when it is absent?

Every call answers only one target question. Reference questions retain their original options and are not answered. Compare target alone versus target + 7 references, each with/without video. Target IDs, question/option text, reference grouping/order, system prompt and generation policy are frozen. Only the video media part is removed; original video-referencing text remains.

## Status and results

- [Completed 10-video / 320-target results](../../results/hd_epic_argon_reference_video_control_20261112/subset10/README.md): 640 no-video successes, zero invalid; one archived timeout with unknown billing. Existing video answers reused.
- [Full73 progress snapshot](../../results/hd_epic_argon_reference_video_control_20261112/full73/progress_snapshot.json): **in progress**, not a completed full-cohort result. Scope: same73 videos,2336 localization targets as the original full context experiment,4672 no-video successes planned.640 exact completed calls retained;4032 new calls. Snapshot is not a live monitor.
- [Exact generation settings](configuration.json), [readable prompt examples](prompt_examples.json).
- Frozen inputs and separate evaluation GT: `full73/frozen/` and `subset10/frozen/`. Gzip files preserve original bytes; hashes are recorded in their plan.json. Credentials and video media are excluded.

## Offline review and reproduction

From the repository root (Python3.9+; native inference scripts use standard-library modules):

```bash
python scripts/materialize_argon_reference_study.py \
  --scope subset10 --destination outputs/argon_reference_subset10_review
python scripts/analyze_argon_reference_novideo.py \
  --root outputs/argon_reference_subset10_review --scope full
```

This reconstructs the 640 saved responses and recomputes the completed 320-target result without API requests. To inspect the full73 inputs and exact retained outcomes:

```bash
python scripts/materialize_argon_reference_study.py \
  --scope full73 --destination outputs/argon_reference_full73_review
python scripts/run_argon_reference_novideo_full73.py \
  --root outputs/argon_reference_full73_review
```

The runner defaults to **offline validation**. Adding `--execute` sends paid requests. Set `GEMINI_API_KEY` in the process environment if separately authorised to execute. Do not launch a duplicate full run while the owner's expansion is active. No key is included here. `scripts/report_argon_reference_novideo.py` and the full73 counterpart need matplotlib to rebuild HTML/figures.

## Execution and audit

Native Vertex Global, actual response model `gemini-4-argon`; MEDIUM thinking, max output32768, JSON. No audio/video sent. Eight concurrent requests for expansion; estimated$40 guard uses frozen historical rates ($2/M uncached input,$0.10/M cached input,$10/M completion including reasoning). This is not an invoice/current-price guarantee. All successful exact calls are retained. No-response read timeouts may be retried once per call, under a16-failure expansion ceiling; invalid/model/HTTP/uncertain calls require review. Network interruption batches are manually reviewed, archived and separately counted. Failed billing remains unknown.

Primary statistic: D=(video context gain)−(no-video context gain), paired by target; recording-cluster bootstrap10000 draws, seed20261112. No multiple-comparison correction. Historical with-video data were not collected contemporaneously; a matching model alias cannot rule out provider updates. Context also changes instruction framing and target position. A zero-crossing interval is not evidence of equivalence, and positive D would not directly measure visual attention. Full historical predictions are included so the finished73-video result can be paired correctly.

Core runner safety, timeout policy and source/reuse gates received independent read-only review. Exact successful reuse, strict JSON, model/usage checks, zero-usage rejection, duplicate-key rejection, failure stops and cost reserves have offline tests. The complete320-target result additionally received an independent raw-response/score/bootstrap/usage audit. No completed full73 effect estimate is claimed by this snapshot.

The frozen full73 plan preserves its preparation-time `historical_paired_context_status`. Retrieval is now complete: the included `historical_full_paired_comparison.csv.gz` and retrieval audit supersede that preparation-time status.
