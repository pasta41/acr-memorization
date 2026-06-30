#!/usr/bin/env bash
#SBATCH --job-name=acr-pythia12b-cq4
#SBATCH --output=acr-pythia12b-cq4.log
#SBATCH --error=acr-pythia12b-cq4.err
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

# Probabilistic ACR (success = P(suffix|prompt) >= b**T), b-sweep, on Pythia-12B base.
# custom_quotes idx 4: "We can't be satisfied just because we're here."
DATA_IDX="${DATA_IDX:-4}"
DATASET="${DATASET:-custom_quotes}"
BVALS="${BVALS:-[0.75,0.775,0.8,0.825,0.85]}"   # b-sweep in ONE model load; override e.g. BVALS=[0.9]

python prompt-minimization-main.py \
  --config-name promptmin_pythia12b_famousquotes \
  dataset="${DATASET}" \
  data_idx="${DATA_IDX}" \
  b_values="${BVALS}"
