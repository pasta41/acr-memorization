#!/usr/bin/env bash
#SBATCH --job-name=acr-pythia12b-cq2
#SBATCH --output=acr-pythia12b-cq2.log
#SBATCH --error=acr-pythia12b-cq2.err
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

# Probabilistic ACR (success = P(suffix|prompt) >= b**T), scan n=5..T-1, on Pythia-12B base.
# custom_quotes idx 2: "My confidence comes from my work ethic. All I can think of is all the hours in the summer."
DATA_IDX="${DATA_IDX:-2}"
DATASET="${DATASET:-custom_quotes}"
BVALS="${BVALS:-[0.7,0.75]}"          # b-sweep, low end = easiest success; override e.g. BVALS=[0.9]
NSTEPS="${NSTEPS:-1000}"              # GCG steps/length; raise to push more successes

python prompt-minimization-main.py \
  --config-name promptmin_pythia12b_famousquotes \
  dataset="${DATASET}" \
  data_idx="${DATA_IDX}" \
  num_steps="${NSTEPS}" \
  b_values="${BVALS}"
