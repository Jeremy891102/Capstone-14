#!/bin/bash
# Qwen3-VL-8B CC4D run on one HPC Cloud Bursting node (2x A100 40GB, TP=2).
# Submit from a Jupyter terminal in OOD:  sbatch scripts/hpc_qwen8b.sbatch
#SBATCH --account=ds_ga_1006_001-2026fa
#SBATCH --partition=c24m170-a100-2
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --job-name=qwen8b-cc4d
#SBATCH --output=logs/qwen8b_%j.log
set -euo pipefail

VIDEO_ROOT=${VIDEO_ROOT:-$HOME/gopro_360p}   # copy the CC4D clips here first
BUDGET=${BUDGET:-250000}                     # video tokens; lower it if vLLM reports OOM / KV too small
OUT=experiments/cc4d_qwen8b

# ponytail: assumes the repo's .venv (vllm + video_report) is already built inside your Singularity/conda env
vllm serve Qwen/Qwen3-VL-8B-Instruct --served-model-name qwen3-vl-8b \
  --tensor-parallel-size 2 --max-model-len 262144 --gpu-memory-utilization 0.92 \
  --media-io-kwargs '{"video": {"fps": 2.0, "num_frames": -1}}' \
  --limit-mm-per-prompt '{"video": 1}' --port 8000 &
until curl -sf localhost:8000/v1/models >/dev/null; do sleep 10; done

mkdir -p $OUT && cp -r experiments/cc4d_qwen/prompts $OUT/
python scripts/make_cc4d_configs.py --data data/cc4d_answer_types_v1 --out $OUT \
  --video-root "$VIDEO_ROOT" --provider vllm --model qwen3-vl-8b --token-budget "$BUDGET" --concurrency 2
for c in $OUT/*/config.yaml; do python scripts/run_experiment.py run --config "$c"; done
