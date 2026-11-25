# Oracle clips: separating order errors from visible step errors

## Result

This is an offline re-score of the existing Qwen3-VL-32B runs. No model calls were made. The question is whether the oracle-clip score changes when a step labelled **Order only** is treated as visually correct: a short clip around one step can show whether that step was performed correctly, but cannot establish whether it happened in the proper sequence relative to other steps.

The answer is **no for the oracle run**: balanced accuracy moves from 53.04% to 52.68% (−0.36 percentage points). It rises slightly for the full-video comparisons, from 53.54% to 53.68% at k=8, and from 51.49% to 51.93% at k=all. All changes are small, and the paired 95% intervals include zero.

### Main step-correct results

Bootstrap intervals are 95% percentile intervals over 2,000 recording-level resamples. Each recording’s steps stay together within a resample.

| Run | Accuracy | Says “correct” | Accuracy on GT-correct steps | Accuracy on GT-wrong steps | Balanced accuracy |
|---|---:|---:|---:|---:|---:|
| Oracle clips | 57.73% [54.52, 61.18] | 14.62% [11.84, 17.55] | 18.49% [14.25, 23.25] | 87.59% [84.62, 90.48] | 53.04% [50.72, 55.51] |
| Full video, k=8 | 51.63% [48.31, 54.94] | 63.62% [59.12, 68.19] | 67.64% [62.99, 72.30] | 39.44% [33.53, 45.42] | 53.54% [50.45, 56.47] |
| Full video, k=all | 46.27% [42.53, 50.16] | 88.22% [82.03, 93.54] | 90.02% [84.00, 95.07] | 12.96% [6.46, 20.34] | 51.49% [49.26, 53.66] |

### Re-label Order-only steps as visually correct

| Run | Accuracy | Says “correct” | Accuracy on relabelled-correct steps | Accuracy on remaining-wrong steps | Balanced accuracy |
|---|---:|---:|---:|---:|---:|
| Oracle clips | 49.74% [46.71, 53.18] | 14.62% [11.89, 17.62] | 17.28% [13.56, 21.17] | 88.07% [84.87, 91.09] | 52.68% [50.67, 54.87] |
| Full video, k=8 | 54.78% [51.22, 58.11] | 63.62% [58.91, 68.09] | 66.99% [61.80, 71.92] | 40.37% [34.24, 46.64] | 53.68% [50.09, 57.09] |
| Full video, k=all | 55.10% [51.96, 58.26] | 88.22% [81.85, 93.51] | 90.10% [83.79, 95.22] | 13.76% [7.36, 21.48] | 51.93% [49.53, 54.44] |

“Says correct” is unchanged by relabelling because predictions do not change. The accuracy denominator is still the same 951 questions; only the GT class labels change. The re-labelled positive class has 515 steps (411 originally correct + 104 Order-only), and the remaining negative class has 436 (94 Order + other + 342 without Order).

Paired change in balanced accuracy (visible label minus standard label): oracle **−0.36 pp** [−1.71, +0.95], k=8 **+0.14 pp** [−2.30, +2.30], and k=all **+0.44 pp** [−1.78, +2.18]. None is distinguishable from zero at this sample size.

Paired balanced-accuracy differences (oracle minus full video):

| Comparison | Standard labels | Order-only treated as correct |
|---|---:|---:|
| Oracle − k=8 | −0.50 pp [−4.14, +3.01] | −1.00 pp [−4.76, +2.94] |
| Oracle − k=all | +1.55 pp [−1.29, +4.41] | +0.75 pp [−2.43, +3.57] |

### Oracle performance by step group

| GT group | n | Recordings | Accuracy |
|---|---:|---:|---:|
| Done correctly | 411 | 64 | 18.49% [14.29, 22.70] |
| Order only | 104 | 33 | 85.58% [78.89, 92.23] |
| Order + other | 94 | 30 | 84.04% [76.09, 91.09] |
| Wrong, no Order | 342 | 64 | 89.18% [85.07, 92.82] |

The high accuracy on wrong steps reflects the oracle model often answering `false`; it is poor at recognizing the correctly performed steps. That default-like behavior is why balanced accuracy is much lower than raw accuracy.

### Oracle accuracy on wrong steps containing each error type

Error types are non-exclusive: a step can contribute to more than one row.

| Error type | n wrong steps | Recordings | Accuracy |
|---|---:|---:|---:|
| Temperature | 20 | 18 | 100.00% [100.00, 100.00] |
| Timing | 59 | 39 | 94.92% [86.66, 100.00] |
| Preparation | 137 | 50 | 91.97% [86.78, 96.27] |
| Measurement | 109 | 51 | 87.16% [80.00, 93.21] |
| Order | 198 | 36 | 84.85% [78.84, 90.32] |
| Technique | 168 | 61 | 83.93% [77.25, 90.00] |

Temperature has the highest point estimate, but only 20 labelled steps; Timing is next. These are conditional accuracy estimates, not a ranking with established statistical separation.

### Error-type multi-select field

| Run | Exact match | Exact match with F/G removed | Answer is exactly `[]` |
|---|---:|---:|---:|
| Oracle clips | 21.05% [18.47, 23.63] | 44.53% [41.26, 47.74] | 14.63% [11.90, 17.63] |
| Full video, k=8 | 35.37% [32.30, 38.39] | 51.05% [47.53, 54.51] | 59.58% [54.96, 64.19] |
| Full video, k=all | 40.53% [36.77, 44.15] | 53.16% [49.78, 56.30] | 88.21% [81.62, 93.71] |

F is Order and G is Missing Step. The oracle error-type intersection has 950 rows rather than 951 because its evaluation has no performed-step `step09_error_types` row for `cc4d-18_101`; k=8 and k=all have that row. The comparison follows the requested per-field intersection, so this missing row does not reduce the boolean-field sample.

## Data and validation

- Oracle: 951 performed steps across 64 recordings. GT groups are done correctly 411, Order-only 104, Order + other 94, and wrong without Order 342 (540 wrong total).
- `stepNN_correct` metrics use 951 paired fields from the three runs. `stepNN_error_types` metrics use the 950-row intersection.
- The documented standard accuracy, says-correct, GT-correct accuracy, GT-wrong accuracy, and balanced accuracy match within 0.1 percentage point for all three runs.
- Bootstrap: recording clusters, 2,000 percentile resamples, seed `20261124`.
- Outputs: [Markdown report](../../../results/cc4d_oracle_order_20261125/report.md), [metrics.csv](../../../results/cc4d_oracle_order_20261125/metrics.csv), and [sources.csv](../../../results/cc4d_oracle_order_20261125/sources.csv).

## Reproduction

The analysis script was copied to `/tmp` on VM2; the evaluation runs and source data stayed in place. Exact transfer and run commands:

```bash
scp -i ~/.ssh/nyu_data_key -o IdentitiesOnly=yes \
  /Users/jeremy/Desktop/NYU/Capstone/Capstone-14-oracle-order-scoring/results/cc4d_oracle_order.py \
  nyu-data@132.196.14.114:/tmp/cc4d_oracle_order.py
ssh -i ~/.ssh/nyu_data_key -o IdentitiesOnly=yes nyu-data@132.196.14.114 \
  'cd ~/cc4d_run && \
  ~/envs/capstone-qwen8b/bin/python /tmp/cc4d_oracle_order.py \
    ~/cc4d_run \
    runs_qwen32/oracle_pad5_k2 \
    runs_qwen32/k32_fwd_k8 \
    runs_qwen32/k32_fwd_kall \
    --out ~/cc4d_run/results/oracle_order_20261125'
```

The CSVs were copied back into this branch at `results/cc4d_oracle_order_20261125/`.

Newest evaluation files used:

| Role | Run directory | Evaluation ID |
|---|---|---|
| Oracle | `runs_qwen32/oracle_pad5_k2` | `20261120T160018Z-5f259b` |
| k=8 | `runs_qwen32/k32_fwd_k8` | `20261110T205510Z-1e3b5b` |
| k=all | `runs_qwen32/k32_fwd_kall` | `20261110T205507Z-564858` |
