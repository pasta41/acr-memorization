# Cluster setup (famous-quotes ACR experiments)

Goal: run the famous-quotes ACR experiments in **dedicated conda envs** so the
`base` env is never modified.

Two envs are needed because the models split across two transformers versions:

| env          | transformers | models |
|--------------|--------------|--------|
| `acr`        | 4.38.2 (pinned) | Pythia-12B, Llama-2-13B base |
| `acr-modern` | >= 4.47          | Llama-3.1-8B base, OLMo-2-13B base |

## The one rule that keeps `base` clean

`pip install` only ever touches the **currently-active** env. So:
**always `conda activate <env>` BEFORE running any pip.** `conda create --clone
base` *copies* base into a new env — it does **not** modify base. The slurm
scripts only `conda activate` (no pip), so jobs can never mutate an env.

## 0. Clone the repo (the fork, on the experiment branch)

```bash
cd /home/users/afcoop
git clone git@github.com:pasta41/acr-memorization.git
cd acr-memorization
git checkout topk-reachability
```

## 1. Create the pinned env `acr`  (Pythia-12B, Llama-2-13B)

Cloning `base` inherits a working CUDA-matched torch; we then install only the
pinned transformers/accelerate on top (no torch churn):

```bash
conda create -n acr --clone base          # base is untouched (read-only source)
conda activate acr                         # activate BEFORE pip
pip install "transformers==4.38.2" "accelerate==0.24.1" \
            hydra-core==1.3.2 omegaconf almost-unique-id sentencepiece
```

(If you prefer the repo's full pin set, `pip install -r requirements.txt`
instead — note that pins `torch==2.1.0`, which may reinstall torch.)

## 2. Create the modern env `acr-modern`  (Llama-3.1-8B, OLMo-2-13B)

```bash
conda create -n acr-modern --clone base   # base is untouched
conda activate acr-modern                  # activate BEFORE pip
pip install -r requirements-modern.txt     # does not reinstall torch
```

## 3. Queue the runs with slurm

Each script requests one A100 (`--gres=gpu:1` → one visible GPU → no sharding),
activates the right env, and runs their unmodified greedy/argmax ACR pipeline.
Default quote is idx 52 (Gretzky); override with `DATA_IDX`.

```bash
# pinned env (acr):
sbatch slurm/run-pythia12b.sh
sbatch slurm/run-llama2-13b.sh

# modern env (acr-modern):
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
