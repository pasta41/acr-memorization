"""
test_topk_reachability.py  (scratch / experiment — NOT part of the upstream repo)

First-pass test of relaxing ACR's success criterion from teacher-forced *argmax*
match to teacher-forced *top-k reachability*: at each target position, is the
gold target token among the model's top-k next-token logits (given the true
preceding tokens)?

This is the minimal, no-optimization version: we teacher-force each target on
its OWN prefix (BOS + target tokens), exactly the discoverable-memorization
conditioning, and check top-k membership position-by-position. It does NOT yet
optimize a prompt (no GCG / free tokens) -- that's the next step, wiring this
predicate into prep_text + the MiniPrompt loop.

Sanity check: with k=1 this reduces to their argmax equality
(utils.check_output_with_hard_tokens), so the k=1 column is the exact-greedy
baseline.

Run:
    python test_topk_reachability.py --model EleutherAI/pythia-160m --n 10
"""
import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def topk_reachability(model, target_ids, ks, device):
    """target_ids: 1D LongTensor of the target token ids (no BOS).

    Teacher-force [BOS, target] and, at each target position t, ask whether
    target_ids[t] is among the top-max(ks) next-token logits predicted from the
    true preceding tokens. Returns {k: bool_tensor[T]} of per-position hits.

    Positions: with input [BOS, y_0, y_1, ..., y_{T-1}], the logits at index i
    predict token i+1. So y_t (input index t+1) is predicted by logits at index
    t. We check logits[0 .. T-1] against targets y_0 .. y_{T-1}.
    """
    bos = model.config.bos_token_id
    if bos is None:
        bos = target_ids[0].item()  # fallback; Pythia/NeoX has a bos
    input_ids = torch.cat(
        [torch.tensor([bos], device=device), target_ids.to(device)]
    ).unsqueeze(0)

    logits = model(input_ids).logits[0]          # [1+T, vocab]
    T = target_ids.size(0)
    pred_logits = logits[0:T]                     # predicts y_0 .. y_{T-1}
    targets = target_ids.to(device)              # [T]

    kmax = max(ks)
    topk_idx = pred_logits.topk(kmax, dim=-1).indices   # [T, kmax]
    # rank of the gold token within the top-kmax, or kmax if outside
    match = topk_idx == targets.unsqueeze(-1)           # [T, kmax]
    hits = {}
    for k in ks:
        hits[k] = match[:, :k].any(dim=-1).cpu()        # [T] bool
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-160m")
    ap.add_argument("--dataset", default="datasets/famous_quotes.json")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--ks", default="1,5,10,50,100")
    args = ap.parse_args()

    device = pick_device()
    ks = [int(x) for x in args.ks.split(",")]
    print(f"device={device}  model={args.model}  ks={ks}\n")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device).eval()

    targets = json.load(open(args.dataset))[: args.n]

    header = "frac of target positions reachable in top-k  (k=1 == argmax / exact-greedy)"
    print(header)
    print("-" * len(header))
    cols = "  ".join(f"k={k:>3}" for k in ks)
    print(f"{'len':>4}  {cols}   target")
    # accumulate fully-reachable rates per k
    full = {k: 0 for k in ks}
    for t in targets:
        ids = tok.encode(t, add_special_tokens=False, return_tensors="pt")[0]
        hits = topk_reachability(model, ids, ks, device)
        fracs = {k: hits[k].float().mean().item() for k in ks}
        for k in ks:
            full[k] += int(hits[k].all().item())
        frac_str = "  ".join(f"{fracs[k]:5.2f}" for k in ks)
        snippet = (t[:50] + "…") if len(t) > 50 else t
        print(f"{ids.size(0):>4}  {frac_str}   {snippet!r}")

    print(f"\nfully-reachable (ALL positions in top-k) out of {len(targets)} targets:")
    for k in ks:
        print(f"  k={k:>3}: {full[k]}/{len(targets)}")


if __name__ == "__main__":
    main()
