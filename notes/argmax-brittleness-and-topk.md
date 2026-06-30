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
