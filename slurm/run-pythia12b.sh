#!/usr/bin/env bash
#SBATCH --job-name=acr-pythia12b
#SBATCH --output=acr-pythia12b.log
#SBATCH --error=acr-pythia12b.err
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

# Famous-quotes ACR, probabilistic relaxation (success = P(suffix|prompt) >= b**T) on
# Pythia-12B base. --gres=gpu:1 -> one visible GPU -> no device_map sharding.
# Override the quote with: DATA_IDX=N sbatch ... ; default 52 = Gretzky.
DATA_IDX="${DATA_IDX:-52}"
B="${B:-0.9}"           # success threshold b: P(target|prompt) >= b**T (sweep with B=0.95 sbatch ...)
DATASET="${DATASET:-famous_quotes}"   # override DATASET=custom_quotes for the separate quote set

python prompt-minimization-main.py \
  --config-name promptmin_pythia12b_famousquotes \
  dataset="${DATASET}" \
  data_idx="${DATA_IDX}" \
  b="${B}"
