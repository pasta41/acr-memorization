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

## The relaxation: reachability *with probability*, not brittle argmax

Coarsest → most faithful; pick deliberately:

1. **Per-position top-k (teacher-forced)** — is the target token in the top-k at each step given the
   true prefix? (What `test_topk_reachability.py` does now.) Widens the margin (top-k vs top-1) so
   less brittle, but still a threshold, still teacher-forced, and a coarse proxy: every position can
   be in-top-k while the *joint* probability is tiny.
2. **Sequence probability** `P(y | prompt) = ∏ₜ p(yₜ | prompt, y_{<ₜ})` — smooth, threshold-free, the
   honest "reachable with probability p". bf16 jitter that flips an argmax barely moves a
   probability. This is the brittleness-free quantity.
3. **Top-k-restricted reachable mass (k-CBS LB)** — does the renormalized top-k tree actually contain
   the target as a reachable path, and with how much mass? Generation-faithful (accounts for the
   autoregressive feedback #1 ignores); gives LB/UB.

## Bridge to the probabilistic-extraction / k-CBS framework

ACR/GCG is the **brittle argmax point-estimate** of reachability. The k-CBS framework is the
**probabilistic, bounded** version of the same question. The Gretzky case is a clean motivating
example: ACR says `success` on a knife-edge prompt that is longer than the quote and doesn't
generate it; a reachability-with-probability measure would instead report the target's probability /
top-k reachable mass — graded, robust, and honest about the brittleness ACR hides.

## Next experiments

1. Extend `test_topk_reachability.py` to report, per target position, the **target's probability and
   rank** (not just in/out of top-k) — and the **joint sequence probability** — evaluated on the
   *GCG-found prompt* (not just the self-prefix). Should make the knife-edge visible (rank-1 by a
   ~1e-5 margin).
2. Compute `P(target | GCG-prompt)` and compare argmax-success vs probability-mass across the four
   models (Pythia-12B, Llama-2-13B, Llama-3.1-8B, OLMo-2-13B). One figure = critique + method.
3. Full version: run the target through k-CBS from the GCG prompt → top-k reachable mass with bounds
   (reframes ACR elicitation as the near-verbatim-mass question).
