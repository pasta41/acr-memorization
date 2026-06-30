#!/usr/bin/env bash
#SBATCH --job-name=acr-llama2-13b-cq1
#SBATCH --output=acr-llama2-13b-cq1.log
#SBATCH --error=acr-llama2-13b-cq1.err
#SBATCH --partition=deho
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH --time=08:00:00

# --- confirm these two lines match the cluster ---
source /home/groups/deho/afcoop/miniconda3/etc/profile.d/conda.sh
conda activate acr                          # single env: transformers>=4.47 covers all 4 models
cd /home/users/afcoop/acr-memorization      # where you cloned the fork (branch: topk-reachability)
# -------------------------------------------------

# Probabilistic ACR (success = P(suffix|prompt) >= b**T), scan n=5..T-1, on Llama-2-13B base.  Gated on HF (Llama-2 token).
# custom_quotes idx 1: "You're allowed to think about the worst possible scenario, but then you have to go do something about it."
DATA_IDX="${DATA_IDX:-1}"
DATASET="${DATASET:-custom_quotes}"
BVALS="${BVALS:-[0.7,0.75]}"          # b-sweep, low end = easiest success; override e.g. BVALS=[0.9]
NSTEPS="${NSTEPS:-1000}"              # GCG steps/length; raise to push more successes

python prompt-minimization-main.py \
  --config-name promptmin_llama2_13b_famousquotes \
  dataset="${DATASET}" \
  data_idx="${DATA_IDX}" \
  num_steps="${NSTEPS}" \
  b_values="${BVALS}"
