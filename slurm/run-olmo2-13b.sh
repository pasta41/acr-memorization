#!/usr/bin/env bash
#SBATCH --job-name=acr-olmo2-13b
#SBATCH --output=acr-olmo2-13b.log
#SBATCH --error=acr-olmo2-13b.err
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
# OLMo-2-13B BASE. OLMo2 architecture needs transformers>=4.47 -> acr-modern env.
# Fully-open model (Dolma training data is public).
DATA_IDX="${DATA_IDX:-52}"

python prompt-minimization-main.py \
  --config-name promptmin_olmo2_13b_famousquotes \
  data_idx="${DATA_IDX}"
