# HD-EPIC Argon: reference context × video control

**Question:** Does adding seven reference questions help a target answer more when video is available than when it is absent?

Every call answers only one target question. Reference questions retain their original options and are not answered. Compare target alone versus target + 7 references, each with/without video. Target IDs, question/option text, reference grouping/order, system prompt and generation policy are frozen. Only the video media part is removed; original video-referencing text remains.

## Status and results

- [Completed 10-video / 320-target results](../../results/hd_epic_argon_reference_video_control_20261112/subset10/README.md): 640 no-video successes, zero invalid; one archived timeout with unknown billing. Existing video answers reused.
- [Completed full73 results](../../results/hd_epic_argon_reference_video_control_20261112/full73/README.md): same 73 videos / 2,336 localization targets; all 4,672 no-video calls succeeded, zero invalid. 640 calls retained; 4,032 newly completed.
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

This reconstructs the 640 saved responses and recomputes the completed 320-target result without API requests. To reconstruct all 4,672 completed full73 responses and validate inputs:

```bash
python scripts/materialize_argon_reference_study.py \
  --scope full73 --destination outputs/argon_reference_full73_review
python scripts/run_argon_reference_novideo_full73.py \
  --root outputs/argon_reference_full73_review
```

Run `python scripts/analyze_argon_reference_novideo_full73.py --root outputs/argon_reference_full73_review` to recompute the full paired analysis. The runner defaults to **offline validation**. Adding `--execute` sends paid requests. Set `GEMINI_API_KEY` in the process environment if separately authorised to execute. The published full73 outputs are complete; offline reconstruction can reuse all successful responses. No key is included here. `scripts/report_argon_reference_novideo.py` and the full73 counterpart need matplotlib to rebuild HTML/figures.

## Execution and audit

Native Vertex Global, actual response model `gemini-4-argon`; MEDIUM thinking, max output32768, JSON. No audio/video sent. Eight concurrent requests for expansion; estimated$40 guard uses frozen historical rates ($2/M uncached input,$0.10/M cached input,$10/M completion including reasoning). This is not an invoice/current-price guarantee. All successful exact calls are retained. No-response read timeouts may be retried once per call, under the automatic supervisor’s 16-failure ceiling; invalid/model/HTTP/uncertain calls require review. After the automatic ceiling stops, transport batches require manual review; both the network interruption batch and a later timeout batch were reviewed and archived. Final expansion has 17 failed attempts plus one imported failure; the automatic ceiling was not increased. Failed billing remains unknown.

Primary statistic: D=(video context gain)−(no-video context gain), paired by target; recording-cluster bootstrap10000 draws, seed20261112. No multiple-comparison correction. Historical with-video data were not collected contemporaneously; a matching model alias cannot rule out provider updates. Context also changes instruction framing and target position. A zero-crossing interval is not evidence of equivalence, and positive D would not directly measure visual attention. Full historical predictions are included so the finished73-video result can be paired correctly.

Core runner safety, timeout policy and source/reuse gates received independent read-only review. Exact successful reuse, strict JSON, model/usage checks, zero-usage rejection, duplicate-key rejection, failure stops and cost reserves have offline tests. The complete320-target result additionally received an independent raw-response/score/bootstrap/usage audit. The completed full73 result and its paired intervals are now included.

The frozen full73 plan preserves its preparation-time `historical_paired_context_status`. Retrieval is now complete: the included `historical_full_paired_comparison.csv.gz` and retrieval audit supersede that preparation-time status.
