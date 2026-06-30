import json
import logging
import os

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

import prompt_optimization as prompt_opt
from prompt_optimization.utils import get_id_func, now, load_target_str

OmegaConf.register_new_resolver("generate_id", get_id_func())


@hydra.main(version_base=None, config_path="config", config_name="promptmin")
def main(cfg):
    # Set randomness
    if cfg.seed:
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    for arg, value in OmegaConf.to_container(cfg, resolve=True).items():
        logging.info(f"{arg}: {value}")

    # Device, model, and tokenizer setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.device_count() > 1:
        model_args = dict(trust_remote_code=True, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16, device_map="auto")
    else:
        model_args = dict(trust_remote_code=True, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16)
    # model_args = dict(trust_remote_code=True, low_cpu_mem_usage=True, torch_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **model_args)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if torch.cuda.device_count() <= 1:
        model = model.to(device)

    if cfg.random_weights:
        logging.info("Randomizing weights")
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0)

    # Data setup
    input_str = cfg.input_str
    target_str = cfg.target_str
    chat_template = cfg.chat_template
    system_prompt = cfg.system_prompt

    if cfg.dataset is not None and cfg.data_idx is not None:
        target_str = load_target_str(cfg.dataset, cfg.data_idx, tokenizer)
        cfg.target_str = target_str
        logging.info(f"Target string selected from dataset, cfg.targer_str: {cfg.target_str}")

    # b values to evaluate. A single model load can sweep several thresholds: set b_values
    # (a list) to sweep, else fall back to the scalar b. Each b is an independent search.
    b_values = cfg.get("b_values", None)
    if b_values is None:
        b_values = [cfg.get("b", 0.9)]
    else:
        b_values = list(b_values)
    base_num_steps = cfg.num_steps          # minimize_prompt mutates num_steps (1.2x ramp) -> reset per b
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    def run_one_b(b):
        # Fresh optimization_args each b (num_steps reset so the ramp doesn't carry over).
        optimization_args = {"discrete_optimizer": cfg.discrete_optimizer,
                             "num_steps": base_num_steps,
                             "lr": cfg.lr,
                             "optimizer": cfg.optimizer,
                             "batch_size": cfg.batch_size,
                             "mini_batch_size": cfg.mini_batch_size,
                             "topk": cfg.topk,
                             "b": b}  # probabilistic success threshold: P(target|prompt) >= b**T
        solution = prompt_opt.minimize_prompt(model, tokenizer, input_str, target_str, system_prompt,
                                              chat_template, device, optimization_args, max_tokens=cfg.max_tokens)
        input_slice, target_slice, loss_slice, input_ids = (solution["input_slice"], solution["target_slice"],
                                                            solution["loss_slice"], solution["input_ids"])

        if solution["success"] is True:
            logging.info("Hard tokens returned:")
            optimized_ids = solution["input_ids"]
            output = model.generate(input_ids=optimized_ids[input_slice].unsqueeze(0), max_new_tokens=20,
                                    do_sample=False)
            optimal_prompt = tokenizer.decode(optimized_ids[input_slice], skip_special_tokens=True)
            logging.info(f"solution: {optimal_prompt}")
            logging.info(f"goal: {tokenizer.decode(input_ids[target_slice], skip_special_tokens=True)}")
            logging.info(f"output: {tokenizer.decode(output[0, target_slice], skip_special_tokens=True)}")

            # Adjusted (probabilistic) ACR = T / |x*|. With |x*| < T (the search only keeps
            # successes shorter than the target), any success here is memorization (ACR > 1).
            target_length = target_slice.stop - target_slice.start
            adjusted_acr = target_length / solution["num_free_tokens"]
            logging.info(f"adjusted ACR = T/|x*| = {target_length}/{solution['num_free_tokens']} = "
                         f"{adjusted_acr:.4f}  (b={b}, P(suffix|prompt)={solution.get('suffix_prob')}, "
                         f"mean_ce={solution.get('mean_ce')})")

            with torch.no_grad():
                ids_t = input_ids[target_slice].unsqueeze(0).to(device)
                loss_of_target_str = model(ids_t, labels=ids_t).loss.item()
            with torch.no_grad():
                ids_p = input_ids[input_slice].unsqueeze(0).to(device)
                loss_of_prompt = model(ids_p, labels=ids_p).loss.item()

            solution["input_ids"] = input_ids.tolist()
            results = {
                "target_length": target_length,
                "acr": adjusted_acr,
                "target_str": target_str,
                "loss_of_target_str": loss_of_target_str,
                "loss_of_prompt": loss_of_prompt,
                "success": True,
                "optimal_prompt": optimal_prompt,
            }
            for k, v in solution.items():
                results[k] = (v.start, v.stop) if isinstance(v, slice) else v
        else:
            logging.info(f"NOT memorized at b={b}: no prompt with < T={solution.get('target_token_count')} "
                         f"tokens reached P(suffix|prompt) >= b**T (ACR <= 1).")
            results = {"success": False, "acr": None, "num_free_tokens": solution["num_free_tokens"],
                       "target_str": target_str, "target_length": target_slice.stop - target_slice.start,
                       "b": solution.get("b"), "target_token_count": solution.get("target_token_count")}

        for k, v in cfg_dict.items():
            results[f"cfg_{k}"] = v
        results["b"] = b            # the actual threshold used for this sweep point
        results["time"] = now()
        for key, value in results.items():
            logging.info(f"{key}: {value}")
        return results

    all_results = []
    for b in b_values:
        logging.info(f"\n================= b = {b} =================")
        all_results.append(run_one_b(b))

    run_dir = HydraConfig.get().run.dir
    if len(all_results) == 1:
        # single b: keep the original results.json (one dict)
        with open(os.path.join(run_dir, "results.json"), "w") as json_file:
            json.dump(all_results[0], json_file)
    else:
        # sweep: one file, a list of per-b result dicts
        with open(os.path.join(run_dir, "results_sweep.json"), "w") as json_file:
            json.dump(all_results, json_file)


if __name__ == "__main__":
    main()
