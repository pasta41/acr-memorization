# Argmax brittleness in ACR/MiniPrompt → motivation for a top-k / probabilistic reachability relaxation

Working notes (branch: `topk-reachability`). Companion to the synthesized critique in
`probabilistic-extraction/notes/acr-memorization-critique.md`. Goal: record what running their
unmodified pipeline revealed, and why it motivates a probabilistic reachability measure.

---

## The motivating result — Pythia-12B, Gretzky quote (`famous_quotes` idx 52)

Target: `You miss 100% of the shots you don't take.` Run: their unmodified greedy/argmax pipeline,
`promptmin_pythia12b_famousquotes`, base model (no chat template), `num_steps=200`, seed 42.

Key fields from `results.json` / log:
- `target_length: 12`  → |y| = 12 tokens
- `num_free_tokens: 15` → |x*| = 15 tokens
- `success: True`
- `optimal_prompt:  sucks}}performance badgeependentiduca myös %. skipping shots ranged Davられ}`
- `goal:   You miss 100% of the shots you don't take.`
- `output: You are also skipping shots.\n\n{/block}`   ← model.generate from the optimal prompt
- `loss_of_target_str: 1.68` (CE of the quote with no prompt — fairly predictable cliché)
- `loss_of_prompt: 13.49` (CE of the adversarial gibberish prompt — very unlikely, as expected)

Two independent "no" signals:
1. **Not memorized.** ACR = |y|/|x*| = 12/15 = **0.8 < 1**. The shortest prompt found is *longer*
   than the quote — the opposite of compression.
2. **Doesn't generate it.** Free-running greedy `generate` from the optimal prompt produces
   "You are also skipping shots…", not the quote — diverges after the 1st token.

…yet the pipeline reports `success: True`.

## What `success` actually means (and doesn't)

`success` is a narrow internal flag: **"GCG found *some* prompt, at *some* length it tried, whose
teacher-forced argmax equals the target at every position."** It is set in `miniprompt.py` via
`check_output_with_hard_tokens` (a single forward over `[prompt+target]`, argmax over the target
positions). It is **not**:
- a memorization verdict — that's ACR > 1, i.e. `num_free_tokens < target_length`, computed
  downstream (here: false);
- a generation check — see below.

So `success: True` + ACR < 1 + generation-mismatch are mutually consistent: a teacher-forced-exact
prompt exists at 15 tokens, but it's longer than the quote and brittle under generation.

## They generate, but ignore it

`prompt-minimization-main.py` lines 80–88: inside `if solution["success"] is True:` they call
`model.generate(..., do_sample=False)` (line 83) and log it as `output:` (line 88). But the
generation is **only logged** — never compared to `goal`, never fed back into `success`, ACR, or
`results.json`. They print a free-running check, it visibly disagrees with the target, and the
pipeline reports `success` anyway. (Flag for the critique.)

## Why argmax success is structurally brittle

Conceptually, teacher-forced argmax-match at every position **is** equivalent to greedy generation
reproducing the target (if argmax==target given the true prefix at every step, greedy decoding
stays on the target). So in *exact arithmetic*, `success` ⟺ greedy reproduction.

The observed `output ≠ goal` is therefore a **numerical artifact, not a conceptual gap**. Leading
hypothesis (NOT yet confirmed — see below):
- The `success` check is one full forward over 27 tokens; `generate` decodes incrementally with a
  KV cache. In bf16 these paths give slightly different logits (FP non-associativity, different
  attention kernels with/without cache).
- GCG stops the instant `match.all()`, so its solutions sit on a **razor-thin argmax margin** — the
  most fragile possible case. A tiny bf16 difference flips one token (here the 2nd), then the
  contexts diverge and the rest cascades.

The deeper point: **argmax success is a measure-zero, knife-edge event.** GCG optimizes the prompt to
land exactly on that edge; any perturbation falls off. A binary argmax criterion is *intrinsically*
fragile — not a bug to tune away. This is the core motivation for a probabilistic notion.

### To confirm the bf16 hypothesis (TODO)
- Re-run the `generate` (or the success forward) in fp32 and check whether `output` then matches.
- Inspect the per-position logit margin between the target token and the runner-up at the divergence
  position — expect a near-tie.

## Caveat on |x*| = 15 (their length search is crude)

The outer loop does +5 on failure, −1 once on success. This run almost certainly went
5 (fail) → 10 (fail) → 15 (success) → 14 (fail) → stop, so it **never tested 11/12/13**. Thus
`|x*| = 15` is a loose *upper bound* (ACR = 0.8 is a *lower bound* on ACR). It found 14 fails, so
it's nowhere near the ≤11 needed for ACR > 1 — "not memorized" is a safe read here — but the exact
number is an artifact of the coarse search + the 200-step (×1.2-on-failure) budget. ACR is monotone
in optimizer effort: more budget can only shorten |x*| / raise ACR (better solver), never the
reverse. Whatever budget is chosen must be applied symmetrically across models and the matched null.

## The relaxation: VERBATIM but PROBABILISTIC (top-k renormalized path probability)

This is **verbatim** (a single fixed exact target — one path), just made **probabilistic**. NOT
near-verbatim: no Levenshtein/Hamming ball, no beam search, **no k-CBS** (that's the near-verbatim
tool). We evaluate the one target path's probability under the **renormalized top-k distribution**.

**The measure.** Fix a `k`. Teacher-force the target `y` given prompt `x`. At each step `t`:
1. take the full next-token distribution `p(· | x, y_{<t})`,
2. keep the **top-k** tokens and **renormalize** them to sum to 1:
   `p̃(v) = p(v) / Σ_{v'∈top-k} p(v')` for `v ∈ top-k`, else `0`,
3. read off `p̃(y_t)` — renormalized prob of the *true* next target token (**0 if `y_t` ∉ top-k**).

Verbatim top-k extraction probability:
```
P_k(y | x) = ∏_t p̃(y_t)
```
i.e. post-process the logits (not argmax) → top-k → renorm → read the target token's prob → multiply.

**Properties:**
- **Generalizes their `success`.** At `k=1`, `P_1 = 1` iff `y` is the argmax at every step (their
  teacher-forced `success`), else `0`. Argmax-success is the brittle `k=1` corner of this.
- **Verbatim** — single fixed path; no ball, no search, no k-CBS.
- **Probabilistic / robust** — graded probability, not a knife-edge; bf16 jitter barely moves `P_k`
  unless a token sits right on the top-k boundary.
- **Interpretation** — `P_k(y|x)` is the probability of emitting `y` *verbatim under top-k sampling*
  from `x`; equivalently, the mass of the single target leaf in the renormalized top-k tree.
  `P_k = 0` cleanly means "not verbatim-reachable under top-k" (the target leaves the top-k somewhere).

(A coarser binary proxy — "is `y_t` in the top-k at each step?" — is what `test_topk_reachability.py`
does now; `P_k` is the graded probability version and is what we actually want.)

## Relation to the probabilistic-extraction framework

ACR/GCG is the **brittle argmax point-estimate** of verbatim reachability. `P_k(y|x)` is the
**probabilistic** version of the *same verbatim question*, using the same renormalized-top-k
probability model as the broader framework — but evaluated on the **single exact target path**
(verbatim), not over a near-verbatim ball (which is the k-CBS regime). The Gretzky case motivates it:
ACR says `success` on a knife-edge `k=1` prompt that is longer than the quote and doesn't even
generate it; `P_k` instead reports a graded verbatim extraction probability under top-k sampling.

## Next experiments

1. Extend `test_topk_reachability.py` to compute **`P_k(y | prompt)`** (top-k renormalized path
   probability) and, per target position, the target token's **renormalized prob + rank**, evaluated
   on the **GCG-found prompt** (not just the self-prefix). Sweep `k`. `k=1` should reproduce the
   argmax `success`; watch how fast `P_k` rises (and the knife-edge: rank-1 by a tiny margin).
2. Compare argmax-`success` vs `P_k` across the four models (Pythia-12B, Llama-2-13B, Llama-3.1-8B,
   OLMo-2-13B) on Gretzky. One figure = the critique + the verbatim-probabilistic measure.
3. (Later) report `P_k` vs `k` curves / pick a principled `k`; consider `P_k` of the target under a
   *natural* prompt, not just the GCG prompt.
