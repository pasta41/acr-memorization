"""
utils.py
functions for preparing text for discrete optimization
"""
import datetime
import json
import math

import torch
import torch.nn.functional as F
from almost_unique_id import generate_id


def load_target_str(dataset_name, idx, tokenizer):
    if dataset_name == "essays":
        with open("datasets/essays_by_avi.json", "r") as fh:
            quote_list = json.load(fh)
            target_str = quote_list[idx]
    elif dataset_name == "famous_quotes":
        with open("datasets/famous_quotes.json", "r") as fh:
            quote_list = json.load(fh)
            target_str = quote_list[idx]
    elif dataset_name == "custom_quotes":
        with open("datasets/custom_quotes.json", "r") as fh:
            quote_list = json.load(fh)
            target_str = quote_list[idx]
    elif dataset_name == "wikipedia":
        with open("datasets/wikipedia.json", "r") as fh:
            quote_list = json.load(fh)
            target_str = quote_list[idx]
    elif dataset_name == "ap":
        with open("datasets/ap-articles-november-2023.json", "r") as fh:
            quote_list = json.load(fh)
            target_str = quote_list[idx]
    elif dataset_name == "random":
        len = 3 + (idx % 15)
        target_ids = torch.randint(0, tokenizer.vocab_size, (100, 20))[idx, :len]
        target_str = tokenizer.decode(target_ids)
    else:
        raise ValueError(f"args.dataset = {dataset_name}, but that option isn't implemented.")
    return target_str


def prep_text(input_str, target_str, tokenizer, system_prompt, chat_template, num_free_tokens, device):
    input_tokens = tokenizer.encode(input_str, return_tensors="pt", add_special_tokens=False).to(device=device)
    target_tokens = tokenizer.encode(target_str, return_tensors="pt", add_special_tokens=False).to(device=device)
    system_prompt_tokens = tokenizer.encode(system_prompt, return_tensors="pt", add_special_tokens=False).to(
        device=device)
    chat_template_tokens = (
        tokenizer.encode(chat_template[0], return_tensors="pt", add_special_tokens=False).to(device=device),
        tokenizer.encode(chat_template[1], return_tensors="pt", add_special_tokens=False).to(device=device))
    free_tokens = torch.randint(0, tokenizer.vocab_size, (1, num_free_tokens)).to(device=device)

    input_ids = torch.cat((chat_template_tokens[0], system_prompt_tokens, input_tokens, free_tokens,
                           chat_template_tokens[1], target_tokens), dim=1).squeeze().long()

    # build slice objects
    tokens_before_free = chat_template_tokens[0].size(-1) + system_prompt_tokens.size(-1) + input_tokens.size(-1)
    free_token_slice = slice(tokens_before_free, tokens_before_free + free_tokens.size(-1))
    input_slice = slice(0, input_ids.size(-1) - target_tokens.size(-1))
    target_slice = slice(input_ids.size(-1) - target_tokens.size(-1), input_ids.size(-1))
    loss_slice = slice(input_ids.size(-1) - target_tokens.size(-1) - 1, input_ids.size(-1) - 1)

    return input_ids, free_token_slice, input_slice, target_slice, loss_slice


def check_output_with_hard_tokens(model, input_ids, target_slice, loss_slice):
    output = model(input_ids)
    match = (output.logits[0, loss_slice].argmax(-1) == input_ids[0, target_slice].squeeze()).all()
    return match


def suffix_logprob(model, input_ids, target_slice, loss_slice):
    """Total teacher-forced log P(target | prompt) over the target positions.

    Base (full) distribution, fp32 log_softmax, summed in log-space over exactly the
    positions GCG scores (loss_slice predicts target_slice). input_ids: (1, seq_len).
    Returns (total_logprob, mean_ce, num_target_tokens), where
    total_logprob = sum_t log p(y_t | prompt, y_<t) and mean_ce = -total_logprob / T.
    """
    with torch.no_grad():
        logits = model(input_ids).logits  # (1, seq_len, vocab)
    # fp32 log_softmax over the full vocab (bf16 loses precision on per-token log-probs)
    log_probs = F.log_softmax(logits[0, loss_slice], dim=-1, dtype=torch.float32)  # (T, vocab)
    targets = input_ids[0, target_slice]  # (T,)
    token_logprobs = log_probs.gather(1, targets.unsqueeze(-1)).squeeze(-1)  # (T,)
    total_logprob = token_logprobs.sum().item()
    num_target_tokens = int(targets.size(0))
    mean_ce = -total_logprob / num_target_tokens
    return total_logprob, mean_ce, num_target_tokens


def check_output_prob_threshold(model, input_ids, target_slice, loss_slice, b):
    """Probabilistic success criterion: P(target | prompt) >= b ** T.

    Equivalent to mean per-token cross-entropy <= -ln(b) (the T's cancel; this is
    exactly a threshold on GCG's own mean-CE loss). Computed under the base
    distribution in fp32. Returns
    (success, prob, mean_ce, total_logprob, num_target_tokens).
    """
    total_logprob, mean_ce, num_target_tokens = suffix_logprob(
        model, input_ids, target_slice, loss_slice
    )
    threshold_logprob = num_target_tokens * math.log(b)  # = ln(b ** T)
    success = bool(total_logprob >= threshold_logprob)
    prob = math.exp(total_logprob)
    return success, prob, mean_ce, total_logprob, num_target_tokens


def now():
    return datetime.datetime.now().strftime("%Y%m%d-%H:%M:%S")


def get_id_func():
    id = generate_id()

    def get_id():
        return id

    return get_id
