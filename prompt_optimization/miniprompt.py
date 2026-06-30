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

    # Memorization is ACR>1 <=> |x*| < T. Goal here is just to SHOW a success exists, and longer
    # prompts (more free tokens) are easiest, so we scan only the top band n = max(5, T-3) .. T-1
    # (ascending) and stop at the FIRST success -- the shortest |x*| within that band. We never test
    # n >= T (ACR <= 1 there). NOTE: restricting to the top band can miss a shorter |x*| below T-3,
    # so the reported ACR is a conservative lower bound (close to 1); widen the band for the true
    # shortest |x*|. Fixed num_steps per length (no adaptive ramp).
    T = target_token_count
    success = False
    best_prompt = None
    best_metrics = None  # (suffix_prob, mean_ce, total_logprob, num_target_tokens)
    x_star = None        # shortest successful prompt length (< T)
    num_steps = optimization_args["num_steps"]

    # Pre-populate slices for the not-memorized fallback (target_slice length == T for any prompt
    # length, which is all main.py needs to report target_length when nothing succeeds).
    n0 = max(1, min(5, T - 1))
    input_ids, free_token_slice, input_slice, target_slice, loss_slice = prompt_opt.prep_text(
        input_str, target_str, tokenizer, system_prompt, chat_template, n0, device)
    best_slices = (free_token_slice, input_slice, target_slice, loss_slice)

    for n in range(max(5, T - 3), T):    # top band: max(5, T-3) .. T-1  (empty if T <= 5)
        logging.info("\n------------------------------------\n")
        logging.info(f"{n} tokens in the prompt")
        input_ids, free_token_slice, input_slice, target_slice, loss_slice = prompt_opt.prep_text(
            input_str, target_str, tokenizer, system_prompt, chat_template, n, device)
        if optimization_args["discrete_optimizer"] == "gcg":
            solution = prompt_opt.optimize_gcg(model, input_ids, input_slice, free_token_slice, target_slice,
                                               loss_slice, num_steps,
                                               batch_size=optimization_args["batch_size"],
                                               topk=optimization_args["topk"],
                                               mini_batch_size=optimization_args["mini_batch_size"],
                                               success_ce_threshold=success_ce_threshold)
        elif optimization_args["discrete_optimizer"] == "random_search":
            solution = prompt_opt.optimize_random_search(model, input_ids, input_slice, free_token_slice,
                                                         target_slice, loss_slice, num_steps,
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
            logging.info(f"Target acquired (P>=b**T) with {n} tokens in the prompt -- shortest |x*|")
            success = True
            x_star = n
            best_prompt = solution["input_ids"]
            best_metrics = (suffix_prob, mean_ce, total_logprob, num_target_tokens)
            best_slices = (free_token_slice, input_slice, target_slice, loss_slice)
            break    # ascending scan: the first success is the shortest prompt -> done
        else:
            logging.info(f"Target NOT acquired with {n} tokens in the prompt")

    output = {"free_token_slice": best_slices[0],
              "input_slice": best_slices[1],
              "target_slice": best_slices[2],
              "loss_slice": best_slices[3],
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
