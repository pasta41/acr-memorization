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
conda activate acr                          # pinned env: transformers==4.38.2 (see requirements.txt)
cd /home/users/afcoop/acr-memorization      # where you cloned the fork (branch: topk-reachability)
# -------------------------------------------------

# Faithful famous-quotes ACR (their unmodified greedy/argmax pipeline) on
# Pythia-12B base. --gres=gpu:1 -> one visible GPU -> no device_map sharding.
# Override the quote with: DATA_IDX=N sbatch ... ; default 52 = Gretzky.
DATA_IDX="${DATA_IDX:-52}"

python prompt-minimization-main.py \
  --config-name promptmin_pythia12b_famousquotes \
  data_idx="${DATA_IDX}"
