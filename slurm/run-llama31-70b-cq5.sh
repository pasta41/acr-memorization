#!/usr/bin/env bash
#SBATCH --job-name=acr-llama31-70b-cq5
#SBATCH --output=acr-llama31-70b-cq5.log
#SBATCH --error=acr-llama31-70b-cq5.err
#SBATCH --partition=deho
#SBATCH --gres=gpu:4
#SBATCH -c 16
#SBATCH --mem=256G
#SBATCH --time=24:00:00

# --- confirm these two lines match the cluster ---
source /home/groups/deho/afcoop/miniconda3/etc/profile.d/conda.sh
conda activate acr                          # single env: transformers>=4.47
cd /home/users/afcoop/acr-memorization      # the fork (branch: topk-reachability)
# -------------------------------------------------

# LARGER-MODEL one-off: Llama-3.1-70B BASE, sharded across 4 GPUs (device_map="auto").
# custom_quotes idx 5: "Everything that is going wrong is controllable."
# NOTE: each job uses ALL 4 GPUs (--gres=gpu:4), so these run SEQUENTIALLY (one at a time),
# and sharded-70B GCG is slow. Defaults kept light (NSTEPS=500, single b=0.7); raise if time allows.
DATA_IDX="${DATA_IDX:-5}"
DATASET="${DATASET:-custom_quotes}"
BVALS="${BVALS:-[0.7]}"               # single b (low end) to keep the 70B one-off cheap
NSTEPS="${NSTEPS:-500}"               # 70B is expensive + memorizes more; raise if time allows

python prompt-minimization-main.py \
  --config-name promptmin_llama31_70b_famousquotes \
  dataset="${DATASET}" \
  data_idx="${DATA_IDX}" \
  num_steps="${NSTEPS}" \
  b_values="${BVALS}"
