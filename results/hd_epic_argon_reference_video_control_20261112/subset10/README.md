# Completed exploratory subset:10 videos,320 identical targets

[English HTML report](report.html) includes figures and embedded CSV/JSON/log downloads. Download and open locally if GitHub displays its source. [Exact paired predictions](paired_predictions.csv), [usage per call](per_call.csv), [summary](summary.json), [raw successful inputs and outputs](experiment_log.jsonl.gz), [independent audit](independent_review.md).

| Condition | Correct /320 | Accuracy |
|---|---:|---:|
| With video, target alone |174|54.38%|
| With video, target +7 references |201|62.81%|
| Without video, target alone |79|24.69%|
| Without video, target +7 references |107|33.44%|

Context gain: with video +8.44pp [5.00,12.50]; without video +8.75pp [3.44,13.44]. Difference D=−0.31pp [−6.56,5.94], paired recording bootstrap10000, seed20261112. These data show context helps without video; they do not show a larger benefit with video. This is not an equivalence result or proof text fully explains the video-condition gain. Video itself still adds about29pp in both protocols.

640 successful no-video calls; no invalid output. One180s read timeout is preserved under attempt_history; only that failed call was retried with identical input.641HTTP attempts, one unknown-billed failure. Successful-call estimate$2.02222 (frozen historical rates, reasoning once);824395input,37343completion including25690reasoning tokens. Median API roundtrip4.94s,p9527.58s, excluding downtime/queue and the failed timeout. Historical video costs excluded.

**These are the completed320-target subset results, not full73-video results.** The full cohort is currently being expanded; see [progress snapshot](../full73/progress_snapshot.json) and [experiment design](../../../experiments/hd_epic_argon_reference_video_control_20261112/README.md). Historical controls are noncontemporaneous and the selected10-video subset is exploratory.
