"""
miniprompt.py
an implementation of miniprompt

developed in collaboration by: Avi Schwarzschild and Zhili Feng and Pratyush Maini in 2024
"""
import logging
import math

import prompt_optimization as prompt_opt


def minimize_prompt(model, tokenizer, input_str, target_str, system_prompt, chat_template, device, optimization_args,
                    max_tokens=30):
    # Probabilistic (verbatim) relaxation of ACR success:
    #   success(prompt) <=> P(target|prompt) >= b**T  <=>  mean per-token CE <= -ln b.
    # This is exactly a threshold on GCG's own loss; b=0.9 by default. See
    # notes/argmax-brittleness-and-topk.md for the derivation and design.
    b = optimization_args.get("b", 0.9)
    success_ce_threshold = -math.log(b)

    # Target length T (tokens). prep_text uses add_special_tokens=False, so this matches
    # the number of target positions scored (target_slice). Used for the b**T threshold
    # and for the early-quit cap (we only care about memorization, ACR>1 <=> |x*| < T).
    target_token_count = len(tokenizer.encode(target_str, add_special_tokens=False))

    # Memorization is ACR>1 <=> |x*| < T. Search strategy (kept deliberately simple):
    # Phase 1 (grow): step up by 5 from n0 (the original coarse search) until we test a size >= T.
    # Phase 2 (decrease): from that overshoot, count DOWN by 1 through EVERY length, stopping at
    #   grow_floor (the largest grow length that failed). This wastes the few sizes >= T but cannot
    #   skip any length < T (the +5 grow step can leave a gap just below T). On a success we keep
    #   decreasing to find the shortest |x*|. ACR = T/|x*| is computed downstream (memorization iff
    #   |x*| < T; a success only at sizes >= T => ACR <= 1 => not memorized).
    T = target_token_count
    success = False
    best_prompt = None
    best_metrics = None  # (suffix_prob, mean_ce, total_logprob, num_target_tokens)
    best_slices = (None, None, None, None)
    x_star = None        # shortest successful prompt length (< T) found
    free_token_slice = input_slice = target_slice = loss_slice = None

    if T <= 1:
        # Degenerate: no prompt can be shorter than a 1-token target -> not memorizable.
        return {"free_token_slice": None, "input_slice": None, "target_slice": None,
                "loss_slice": None, "success": False, "num_free_tokens": None,
                "input_ids": None, "b": b, "target_token_count": T,
                "suffix_prob": None, "mean_ce": None, "total_logprob": None}

    n = min(5, T - 1)
    grow_floor = 0       # lengths <= grow_floor are known-fail (from the +5 grow phase)
    descending = False   # False during the +5 grow phase; True while counting down by 1

    while True:
        logging.info("\n------------------------------------\n")
        logging.info(f"{n} tokens in the prompt")
        input_ids, free_token_slice, input_slice, target_slice, loss_slice = prompt_opt.prep_text(
            input_str, target_str, tokenizer, system_prompt, chat_template, n, device)
        if optimization_args["discrete_optimizer"] == "gcg":
            solution = prompt_opt.optimize_gcg(model, input_ids, input_slice, free_token_slice, target_slice,
                                               loss_slice, optimization_args["num_steps"],
                                               batch_size=optimization_args["batch_size"],
                                               topk=optimization_args["topk"],
                                               mini_batch_size=optimization_args["mini_batch_size"],
                                               success_ce_threshold=success_ce_threshold)
        elif optimization_args["discrete_optimizer"] == "random_search":
            solution = prompt_opt.optimize_random_search(model, input_ids, input_slice, free_token_slice,
                                                         target_slice, loss_slice, optimization_args["num_steps"],
                                                         batch_size=optimization_args["batch_size"],
                                                         mini_batch_size=optimization_args["mini_batch_size"])
        else:
            raise ValueError(
                "discrete_optimizer must be one of ['gcg', 'random_search']")

        target_acquired, suffix_prob, mean_ce, total_logprob, num_target_tokens = \
            prompt_opt.check_output_prob_threshold(model, solution["input_ids"].unsqueeze(0),
                                                   target_slice, loss_slice, b)
        logging.info(f"P(suffix|prompt)={suffix_prob:.3e} | mean_ce={mean_ce:.4f} | "
                     f"threshold b**T={b ** num_target_tokens:.3e} (b={b}, T={num_target_tokens}) | "
                     f"acquired={target_acquired}")

        if target_acquired:
            logging.info(f"Target acquired (P>=b**T) with {n} tokens in the prompt")
            success = True
            x_star = n
            best_prompt = solution["input_ids"]
            best_metrics = (suffix_prob, mean_ce, total_logprob, num_target_tokens)
            best_slices = (free_token_slice, input_slice, target_slice, loss_slice)
            descending = True            # found one; count down to look for a shorter prompt
            next_n = n - 1
        else:
            logging.info(f"Target NOT acquired with {n} tokens in the prompt")
            optimization_args["num_steps"] = int(optimization_args["num_steps"] * 1.2)
            if not descending:
                if n >= T:
                    # Just tested the overshoot (size >= T). Now run the full decrease loop from
                    # here down -- wastes the >= T sizes but cannot skip any length < T.
                    descending = True
                    next_n = n - 1
                else:
                    grow_floor = n           # largest < T length that failed during grow
                    next_n = n + 5           # keep growing by 5 (will test the first size >= T)
            else:
                next_n = n - 1               # decrease by 1 through every remaining length

        # Stop once the decrease loop reaches the known-fail floor, or we leave the valid range.
        if descending and next_n <= grow_floor:
            break
        if next_n < 1 or next_n == n:
            break
        n = next_n

    output = {"free_token_slice": best_slices[0] if best_slices[0] is not None else free_token_slice,
              "input_slice": best_slices[1] if best_slices[1] is not None else input_slice,
              "target_slice": best_slices[2] if best_slices[2] is not None else target_slice,
              "loss_slice": best_slices[3] if best_slices[3] is not None else loss_slice,
              "success": success,
              "num_free_tokens": x_star if success else None,
              "input_ids": best_prompt,
              "b": b,
              "target_token_count": T,
              "suffix_prob": best_metrics[0] if best_metrics is not None else None,
              "mean_ce": best_metrics[1] if best_metrics is not None else None,
              "total_logprob": best_metrics[2] if best_metrics is not None else None,
              }
    return output
