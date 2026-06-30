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

    n_tokens_in_prompt = min(5, max(1, target_token_count - 1))
    running_max = max_tokens
    running_min = 0
    success = False
    best_prompt = None
    best_metrics = None  # (suffix_prob, mean_ce, total_logprob, num_target_tokens)
    done = False
    best_slices = (None, None, None, None)

    while not done:
        logging.info("\n------------------------------------\n")
        logging.info(f"{n_tokens_in_prompt} tokens in the prompt")
        input_ids, free_token_slice, input_slice, target_slice, loss_slice = prompt_opt.prep_text(input_str,
                                                                                                  target_str,
                                                                                                  tokenizer,
                                                                                                  system_prompt,
                                                                                                  chat_template,
                                                                                                  n_tokens_in_prompt,
                                                                                                  device)
        if running_max == -1:
            running_max = (target_slice.stop - target_slice.start) * 5
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
            logging.info(f"Target acquired (P>=b**T) with {n_tokens_in_prompt} tokens in the prompt")
            running_max = n_tokens_in_prompt
            success = True
            best_prompt = solution["input_ids"]
            best_metrics = (suffix_prob, mean_ce, total_logprob, num_target_tokens)
            new_num_tokens = n_tokens_in_prompt - 1
            best_slices = (free_token_slice, input_slice, target_slice, loss_slice)
        else:
            logging.info(f"Target NOT acquired with {n_tokens_in_prompt} tokens in the prompt")
            new_num_tokens = n_tokens_in_prompt + 5
            # Early-quit: only memorization (ACR>1 <=> |x*| < T) is of interest. If growth
            # would reach/exceed T, run one final round at T-1 instead, then quit (handled by
            # running_min below). shrink-on-success covers the remaining <T fill-ins.
            if new_num_tokens >= target_token_count:
                new_num_tokens = target_token_count - 1
            running_min = n_tokens_in_prompt
            optimization_args["num_steps"] = int(optimization_args["num_steps"] * 1.2)

        if (new_num_tokens >= running_max) or (new_num_tokens <= running_min):
            done = True
        else:
            n_tokens_in_prompt = new_num_tokens

    output = {"free_token_slice": best_slices[0] if best_slices[0] is not None else free_token_slice,
              "input_slice": best_slices[1] if best_slices[1] is not None else input_slice,
              "target_slice": best_slices[2] if best_slices[2] is not None else target_slice,
              "loss_slice": best_slices[3] if best_slices[3] is not None else loss_slice,
              "success": success,
              "num_free_tokens": running_max,
              "input_ids": best_prompt,
              "b": b,
              "target_token_count": target_token_count,
              "suffix_prob": best_metrics[0] if best_metrics is not None else None,
              "mean_ce": best_metrics[1] if best_metrics is not None else None,
              "total_logprob": best_metrics[2] if best_metrics is not None else None,
              }
    return output
