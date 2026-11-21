# Qwen3-VL-32B attention intervention: does scaling up video attention fix k = all?

Status (2026-11-21): finished on all 54 reports (class A 23, class B 31), k = all, forward order.

Model: Qwen3-VL-32B-Instruct. Data: `cc4d_k32_v1`. Background and the attention probe that motivated this: [attention_probe.md](attention_probe.md) (the late-layer video share is about 13 pp lower at k = all than at k = 8).

## 1. Method and results

Regenerate the k = all answers of all 54 reports (`scripts/qwen_intervene.py`, scored by `results/cc4d_intervene_score.py`). Prefill once, then decode once per condition from the same KV cache. For the generated tokens only, add log α to the attention scores on the target keys before the softmax. The first generated token comes from the prefill and is not modified.

Conditions: `base` (α = 1, same manual code path); `a1.5`, `a2`, `a3` (last quarter of layers, video keys); `all2` (all layers, video keys); `text2` (control: last quarter of layers, prompt-text keys). `base` is the baseline instead of vLLM's saved answers, because the manual attention differs from the fused kernel by bf16 rounding; the two agree on accuracy (33.4% vs 33.5%).

2,557 ordinary questions, 54 reports, 95% bootstrap intervals over reports:

| Condition | Accuracy | Minus base (pp) [95% CI] | Late-layer video share | Answers changed vs base |
| --- | --- | --- | --- | --- |
| base | 33.4% | – | 49.6% | – |
| a1.5 | 33.6% | +0.2 [−0.1, +0.6] | 57.6% | 3.2% |
| a2 | 33.0% | −0.4 [−1.2, +0.2] | 62.8% | 4.7% |
| a3 | 32.7% | −0.6 [−1.6, +0.1] | 69.4% | 6.0% |
| all2 | 33.3% | −0.0 [−0.6, +0.5] | 62.5% | 7.4% |
| text2 (control) | 33.0% | −0.4 [−0.7, −0.0] | 36.0% | 4.4% |
| vLLM k = 8 (reference) | 35.6% | +2.3 [−0.2, +4.7] | – | – |

- The intervention does what it should (video share rises with α, falls for the text control) and the video share at α = 3 is above the k = 8 level (60%), yet accuracy does not move toward k = 8.
- No answer type shows a consistent direction (bool, multi-choice and seconds all within ±1 pp; integer and single-choice have 54 and 39 questions).
- The text control also lowers accuracy slightly, so the small drop at α = 2–3 is as consistent with "perturbing attention hurts a little" as with anything about the video.
- All outputs parsed; output length unchanged.

Reading: the 13 pp attention gap in section 3.2 of [attention_probe.md](attention_probe.md) is not what costs k = all its accuracy, or at least restoring it with one global factor does not help. Limits: one factor for all heads in the late layers (no per-head intervention); a k = 8 gap of 2–4 pp leaves little room to detect a recovery; only α up to 3.

## 2. Reproduction

- `scripts/qwen_intervene.py RUN_DIR OUT.jsonl --model ... --conditions base,a1.5,a2,a3,all2,text2 --mlp-chunk 32768 --max-gpu-mem 22` (class A: `--max-seconds 660`; class B: `--min-seconds 660 --max-seconds 1020`). Resumable. Run on vm2 with `~/envs/capstone-qwen8b`, 4 × A100 80GB, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; queue script `~/intervene_queue.sh`, log `~/intervene_queue.log`.
- `results/cc4d_intervene_score.py RUN_DIR GT.jsonl ROWS.json.gz OUT.json INTERVENE.jsonl ...` scores the outputs with the project's parser and scorer.
- Raw output (vm2 only, not in the repo): `~/cc4d_run/gen/intervene_A.jsonl` (138 records), `intervene_B.jsonl` (186), scores in `gen/intervene_scores.json`. One record per (call, condition): generated text, attention shares, token counts, timings.
