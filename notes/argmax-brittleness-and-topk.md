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

## Four-model Gretzky cross-tab (single-quote illustration)

Same target (`famous_quotes` idx 52), their unmodified pipeline, base models (no chat template),
num_steps=200, seed 42. `generates verbatim?` = does the logged greedy `output` equal the goal.

`|x*|` = `num_free_tokens` = length of the shortest success prompt.

| model | `success` | |y| | \|x*\| | ACR | memorized (ACR>1)? | generates verbatim? |
|---|---|--:|--:|--:|---|---|
| Pythia-12B   | True | 12 | 15 | 0.80 | No  | No  |
| Llama-2-13B  | True | 17 | 9  | 1.89 | Yes | No  |
| Llama-3.1-8B | True | 13 | 34 | 0.38 | No  | Yes |
| OLMo-2-13B   | True | 13 | 9  | 1.44 | Yes | Yes |

**Punchline.** `success` is True for all four and is uninformative: the (memorized?, generates?)
cross-tab hits all four cells. Worse, **ACR and actual generation have no monotone relationship** —
the two models that generate the quote have ACRs 0.38 and 1.44; the two that don't have 0.80 and 1.89.
The off-diagonal cases are damning:
- **Llama-2-13B**: ACR 1.89 (highest, "most memorized") but does **not** generate the quote (greedy → the Elf line).
- **Llama-3.1-8B**: ACR 0.38 (lowest, "least memorized") but **does** generate the quote cleanly.

So on this quote ACR's "memorized" label is *uninformative about — even inverted from* — verbatim
extractability under generation. This is a single-quote *illustration*, not a statistical claim — the
systematic version is the probabilistic measure (below) run over the set. It is, however, the cleanest
motivation for a graded, generation-faithful measure: the binary teacher-forced `success` collapses
four very different situations into one label, and even the ACR>1 threshold dissociates from real
extractability.

Free no-prompt baseline already in `results.json`: `loss_of_target_str` is the quote's unconditional
per-token NLL, so no-prompt geometric-mean per-token prob = exp(-loss): Pythia 0.19, Llama-2 0.39,
Llama-3.1 0.21, OLMo 0.23 — all far below a 0.9 bar unconditionally, so the GCG prompt's job is to lift
P(suffix|prompt); our measure quantifies by how much.

## The no-prompt baseline (`loss_of_target_str`)

`loss_of_target_str` (in every `results.json`) is the model's per-token loss on the **quote alone**,
no prompt. The main script computes it as:
```python
ids = input_ids[target_slice].unsqueeze(0)   # JUST the quote's tokens, nothing in front
outputs = model(ids, labels=ids)
loss_of_target_str = outputs.loss.item()
```
The GCG free tokens are sliced off, and there is no BOS (`prep_text` uses `add_special_tokens=False`).
It's a single teacher-forced pass over the bare quote — *scoring*, not generation.

Token-by-token (HF shifts so `logits[:-1]` predict `labels[1:]`):
- the **first quote token is the seed** — fed at position 0, **not scored** (nothing precedes it);
- `y1` scored as `p(y1 | y0)`, `y2` as `p(y2 | y0 y1)`, …, `y_{T-1}` as `p(y_{T-1} | y0…y_{T-2})`.

So each quote token is conditioned only on the **earlier part of the quote itself**. `exp(-loss)` =
the **geometric-mean per-token probability** of the quote with no external prompt. Observed values
(Gretzky): Pythia 0.19, Llama-2 0.39, Llama-3.1 0.21, OLMo 0.23 — all far below a 0.9 bar
unconditionally. This is the floor the GCG prompt has to lift; `P(suffix|prompt)` measures by how much.

Effectively it's **seeded by the quote's own first token** — a very weak, 1-token, in-distribution
cue — not a true zero-cue measurement.

**Caveat (it is NOT the natural-prefix baseline):** `loss_of_target_str` is the **no-prompt /
self-conditioned** baseline, not the quote-given-a-real-preceding-cue baseline (the discoverable-
extraction setup). Two mismatches make it not directly comparable to our with-prompt `P(suffix|prompt)`:
- it averages over **T−1** tokens (the first quote token contributes no loss), whereas
  `P(suffix|prompt)` scores all **T** suffix tokens;
- its first token is **unconditioned**, whereas ours is conditioned on the prompt.

So treat it as a rough *anchor* (unconditional predictability of the quote), and compute the true
natural-prefix contrast separately if we want an apples-to-apples comparison.

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

## GCG loss ⟺ probability threshold (full derivation, no skipped steps)

The threshold success criterion `P(suffix|prompt) ≥ b^T` is *identically* a threshold on GCG's own
loss. GCG minimizes the **mean per-token cross-entropy** over the T target tokens
(their `F.cross_entropy(..., reduction='mean')`):

    loss = −(1/T) · Σ_{t=1}^{T} log p(y_t | prompt, y_{<t})                      (1)

The teacher-forced extraction probability of the target is the product of the per-token conditionals:

    P(suffix | prompt) = ∏_{t=1}^{T} p(y_t | prompt, y_{<t})
                       = exp( Σ_{t=1}^{T} log p(y_t | prompt, y_{<t}) )          (2)

From (1), multiply both sides by −T:

    Σ_{t=1}^{T} log p(y_t | prompt, y_{<t}) = −T · loss                          (3)

Substitute (3) into (2):

    P(suffix | prompt) = exp(−T · loss)                                          (4)

Now apply the threshold and simplify, one step at a time:

    P(suffix | prompt) ≥ b^T
    exp(−T · loss)     ≥ b^T                  [substitute (4)]
    ln( exp(−T·loss) ) ≥ ln( b^T )            [ln is strictly increasing ⇒ inequality direction preserved]
    −T · loss          ≥ T · ln b             [ln(e^x) = x ; ln(b^T) = T·ln b]
    −loss              ≥ ln b                 [divide both sides by T > 0 ⇒ direction preserved]
    loss               ≤ −ln b                [multiply both sides by −1 ⇒ direction flips]

Therefore:

    P(suffix|prompt) ≥ b^T   ⟺   loss ≤ −ln b                                    (5)

Equivalently, the **geometric-mean per-token probability** is `P^{1/T} = exp(−loss)`, so the same rule
reads `exp(−loss) ≥ b ⟺ loss ≤ −ln b` — "average per-token probability ≥ b."

For b = 0.9: −ln(0.9) ≈ 0.10536 nats, so "success" = GCG's mean-CE loss ≤ 0.105.

The T's cancel ⇒ a **length-independent bound on GCG's loss** (the `b^T` form is exactly what
length-normalizes it, giving a per-token bar comparable across targets of different T). The
geometric-mean form matches GCG's `mean` reduction; a **min-per-token** rule would not (mean reduction
can sacrifice a single token), so min-per-token is a reported *diagnostic*, not the success test.

## Relation to the probabilistic-extraction framework

ACR/GCG is the **brittle argmax point-estimate** of verbatim reachability. `P_k(y|x)` is the
**probabilistic** version of the *same verbatim question*, using the same renormalized-top-k
probability model as the broader framework — but evaluated on the **single exact target path**
(verbatim), not over a near-verbatim ball (which is the k-CBS regime). The Gretzky case motivates it:
ACR says `success` on a knife-edge `k=1` prompt that is longer than the quote and doesn't even
generate it; `P_k` instead reports a graded verbatim extraction probability under top-k sampling.

## Finalized design — step 1 (probabilistic ACR via a loss threshold)

Decisions (this branch):
- **Base distribution** (full softmax, temperature 1), fp32, sum-logprobs-then-exp. No top-k yet
  (top-k renormalized variant is later).
- **Report (graded):** for the success/best prompt — GCG mean-CE `loss` *and* the teacher-forced
  extraction probability `P(suffix|prompt) = exp(−T·loss)`.
- **Binarize (for ACR):** success ⟺ `P(suffix|prompt) ≥ b^T` ⟺ `loss ≤ −ln b`. This binary drives the
  length search and yields an **adjusted ACR = T / |x*|** (|x*| = shortest prompt length achieving
  success). **Print the ACR in the output/log.**
- **Config knob `b`**, default **0.9** (⇒ loss bar ≈ 0.105 nats). Sweepable post-hoc.
- **Keep the length search** (ACR-comparability with the paper).
- **Early-quit (only memorization, ACR>1 ⟺ |x*|<T, matters here):** when grow-on-failure would set the
  next length ≥ T, run **one extra round at T−1** instead, then quit. T−1 fails → "not memorized at b";
  T−1 succeeds → memorized, and the existing shrink-by-1-on-success finds the shortest |x*|. Relies on
  rough monotonicity of success in prompt length; with shrink-on-success this captures all length-<T
  fill-ins the crude +5 step would skip. Wastes ≤ a few boundary rounds — acceptable.

Implementation (three small changes on the branch):
1. `miniprompt.py` success: replace `check_output_with_hard_tokens` with the `loss ≤ −ln b` test (base
   dist, fp32; reuse the `verbatim_probabilistic/scoring.py` logic).
2. `gcg.py` inner early-stop: break when `loss ≤ −ln b` instead of `match.all()`.
3. `miniprompt.py` outer loop: the T−1-then-quit early-exit; thread `b` through the config; log `loss`,
   `P(suffix|prompt)`, and the adjusted ACR.

**Step 2 (separate, later):** raise the GCG step budget (`num_steps`) as its own monotone axis — do NOT
combine with step 1 (avoid confounding the criterion change with the budget change).

**Later:** top-k renormalized `P_k` + per-position rank/margin diagnostics; natural-prefix contrast
(quote given a real cue, vs the adversarial GCG prompt).

## Verification: the paper *defines* greedy decoding; the released code *teacher-forces*

Checked the paper text (arXiv 2404.15146 v2) against the released code, because the gap is significant.
- **Paper definition (§3.1, Alg. 1):** `M(x)=y` means the model *generates* `y` under **greedy decoding**
  (autoregressively appending the argmax). Quote: "M can perform generation by repeatedly predicting the next
  token … with the argmax … appended at each step (this process is called greedy decoding) … we will also call
  the greedy decoding result the output of M." Algorithm 1's success test is written `M(z)=y`.
- **Released code:** `success` = `check_output_with_hard_tokens` = **teacher-forced argmax** over the target
  positions. `prompt-minimization-main.py` calls `model.generate` (line 83) but only **logs** it — never
  compares to the target, never gates `success` / ACR / `results.json`.
- **Equivalence + gap:** teacher-forced argmax-match-everywhere ⟺ greedy decoding reproduces the target *in
  exact arithmetic*; they diverge in bf16 (full-forward check vs KV-cached generate), and GCG sits exactly on
  the argmax margin (stops at first `match.all()`), maximally exposing the gap. So the code's `success` can —
  and on 2/4 models (Pythia-12B, Llama-2-13B) did — mark prompts as memorized that **fail the paper's own
  greedy-decoding definition**.
- **Scope/caveat:** confirmed for the *released code* and *our four runs*. NOT independently confirmed that the
  *published numbers* used this unverified check (they may have re-verified) — but the official pipeline does
  not gate on generation, and the definition requires it.

## Run plan / sequencing
1. **Now:** step-1 relaxation on the **same quote** (Gretzky, idx 52), 4 base models, **current budget (200)**.
2. **Then: other quotes** — TODO, pick a set (canonical short quotes + longer / post-cutoff controls). [Reminder
   to AFC: more quotes to run once Gretzky validates the relaxation.]
3. **Then:** raise the GCG step budget (`num_steps`) as a separate monotone axis (step 2). Don't combine with 1.

## Implementation status (step 1 — DONE + smoke-tested)
- `utils.py`: `suffix_logprob`, `check_output_prob_threshold` (base dist, fp32, sum-logprobs-then-exp).
- `gcg.py`: `success_ce_threshold` inner early-stop (break when mean-CE ≤ −ln b; argmax fallback if None).
- `miniprompt.py`: threshold success, T−1 early-quit cap, probabilistic metrics in output.
- `prompt-minimization-main.py`: thread `b` (cfg, default 0.9), compute/log/save adjusted ACR + metrics.
- Smoke-tested on pythia-160m (CPU, fp32): not-memorized path (b=0.5 → early-quit at T−1) and memorized path
  (b=0.001 → shrink to |x*|=1, ACR=7) both run clean; threshold logging, early-quit, and num_steps ramp verified.
  Not yet exercised on the 4 base-model cluster configs. `b` defaults to 0.9 (config-overridable, or `b=` on CLI).
