# Cluster setup (famous-quotes ACR experiments)

Goal: run the famous-quotes ACR experiments in a **dedicated conda env** so the
`base` env is never modified.

**One env covers all four models.** transformers >= 4.47 supports Pythia,
Llama-2, Llama-3.1, and OLMo-2, so a single `acr` env runs everything. (The
repo's 4.38.2 pin is not required: same weights + deterministic GCG → identical
greedy-ACR results across transformers versions.)

| env   | transformers | models |
|-------|--------------|--------|
| `acr` | >= 4.47 (here: 4.50.0) | Pythia-12B, Llama-2-13B, Llama-3.1-8B, OLMo-2-13B (all base) |

## The one rule that keeps `base` clean

`pip install` only ever touches the **currently-active** env. So:
**always `conda activate acr` BEFORE running any pip.** The slurm scripts only
`conda activate` (no pip), so jobs can never mutate an env.

NOTE: do **not** `conda create --clone base` here — base is the *root* env and
conda refuses to clone its own management packages out of root. Create a **fresh**
env instead (below).

## 0. Clone the repo (the fork, on the experiment branch)

```bash
cd /home/users/afcoop
git clone git@github.com:pasta41/acr-memorization.git
cd acr-memorization
git checkout topk-reachability
```

## 1. Create the env `acr`  (all four models)

Fresh env, pinned to base's proven versions so torch matches the cluster CUDA:

```bash
conda create -n acr python=3.11 -y
conda activate acr                         # activate BEFORE pip
pip install torch==2.6.0 transformers==4.50.0 accelerate==1.6.0 \
            hydra-core==1.3.2 omegaconf almost-unique-id sentencepiece protobuf
```

## 3. Queue the runs with slurm

Each script requests one A100 (`--gres=gpu:1` → one visible GPU → no sharding),
activates the right env, and runs their unmodified greedy/argmax ACR pipeline.
Default quote is idx 52 (Gretzky); override with `DATA_IDX`.

```bash
# all use the single acr env:
sbatch slurm/run-pythia12b.sh
sbatch slurm/run-llama2-13b.sh
sbatch slurm/run-llama31-8b.sh
sbatch slurm/run-olmo2-13b.sh        # secondary

# a different quote:
DATA_IDX=0 sbatch slurm/run-pythia12b.sh
```

Output: `outputs/<experiment_name>/<run_id>/results.json` (optimized prompt,
`success`, target length, losses) plus per-job `acr-*.log` / `acr-*.err`.

## Notes / things to confirm

- The two `# confirm these two lines` at the top of each slurm script: the conda
  path (`/home/groups/deho/afcoop/miniconda3/...`) and repo path
  (`/home/users/afcoop/acr-memorization`) — adjust if different.
- HF access: Llama-2-13B and Llama-3.1-8B are gated — make sure your HF token is
  available on the cluster (`huggingface-cli login` or `HF_TOKEN`).
- OLMo-2 model id `allenai/OLMo-2-1124-13B`: confirm the `-1124-` date stamp on HF.
- These configs run the **unmodified greedy/argmax** pipeline (their original
  ACR). The top-k reachability relaxation lives only in
  `test_topk_reachability.py` and is not used by these runs.
