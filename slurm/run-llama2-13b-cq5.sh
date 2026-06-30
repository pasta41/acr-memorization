#!/usr/bin/env bash
#SBATCH --job-name=acr-llama2-13b-cq5
#SBATCH --output=acr-llama2-13b-cq5.log
#SBATCH --error=acr-llama2-13b-cq5.err
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

# Probabilistic ACR (success = P(suffix|prompt) >= b**T), b-sweep, on Llama-2-13B base.  Gated on HF (Llama-2 token).
# custom_quotes idx 5: "Everything that is going wrong is controllable."
DATA_IDX="${DATA_IDX:-5}"
DATASET="${DATASET:-custom_quotes}"
BVALS="${BVALS:-[0.75,0.775,0.8,0.825,0.85]}"   # b-sweep in ONE model load; override e.g. BVALS=[0.9]

python prompt-minimization-main.py \
  --config-name promptmin_llama2_13b_famousquotes \
  dataset="${DATASET}" \
  data_idx="${DATA_IDX}" \
  b_values="${BVALS}"
