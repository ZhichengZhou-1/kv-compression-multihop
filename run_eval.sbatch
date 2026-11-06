#!/bin/bash
#SBATCH --job-name=kv_multihop
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=default
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out

# Submit from the repo root (logs/ must exist there).
# Usage: sbatch run_eval.sbatch <out.jsonl> <n_per_config> <condition> [condition ...]
# e.g.   sbatch run_eval.sbatch results/baseline.jsonl 25 full nocontext
# Extra eval_kv.py flags go in EXTRA_ARGS, e.g.
#        sbatch --export=ALL,EXTRA_ARGS="--model Qwen/Qwen2.5-7B-Instruct --cot --max-new-tokens 160" run_eval.sbatch ...

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
export HF_HOME=${HF_HOME:-$PWD/hf_cache}

OUT=$1
N=$2
shift 2

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
.venv/bin/python eval_kv.py --data data/two_hop.jsonl --out "$OUT" --n-per-config "$N" ${EXTRA_ARGS:-} --conditions "$@"
