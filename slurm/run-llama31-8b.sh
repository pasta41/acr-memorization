#!/usr/bin/env bash
#SBATCH --job-name=acr-llama31-8b
#SBATCH --output=acr-llama31-8b.log
#SBATCH --error=acr-llama31-8b.err
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

# Faithful famous-quotes ACR (their unmodified greedy/argmax pipeline) on
# Llama-3.1-8B BASE. Needs transformers>=4.43 (Llama-3.1 RoPE scaling) -> the
# acr-modern env. Gated on HF (needs your Llama access token).
DATA_IDX="${DATA_IDX:-52}"

python prompt-minimization-main.py \
  --config-name promptmin_llama31_8b_famousquotes \
  data_idx="${DATA_IDX}"
