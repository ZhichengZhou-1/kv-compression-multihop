"""Evaluate paired single-hop / two-hop questions under KV cache compression.

The document is prefilled once per condition; the cache is then compressed (or not) and all three
questions (hop1, hop2, two_hop) are answered from the *same* cache. After each question the cache is
trimmed back to the document, so questions never see each other.

Conditions:
    full         uncompressed cache
    nocontext    questions asked with an empty document (checks answers need the context)
    rand<r>      kvpress RandomPress, evicting a fraction r of document KV entries (e.g. rand0.5)
    int<b>       per-token round-to-nearest fake quantization of document K/V to b bits (e.g. int4)

Usage:
    python eval_kv.py --data data/two_hop.jsonl --n-per-config 5 --conditions full nocontext
    python eval_kv.py --conditions full nocontext rand0.25 rand0.5 rand0.75 int8 int4 int3 int2
"""

import argparse
import collections
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

INSTRUCTION = "Read the following facts carefully.\n\n"
QUESTION_TEMPLATE = "\n\nQuestion: {q}\nAnswer with only the name, no other words."
COT_QUESTION_TEMPLATE = (
    "\n\nQuestion: {q}\nThink step by step using the facts above, "
    "then give the final answer on the last line as 'Answer: <name>'."
)


def extract_answer(pred, cot):
    """In CoT mode only the text after the last 'Answer:' counts."""
    if cot and "answer:" in pred.lower():
        return pred[pred.lower().rindex("answer:") + len("answer:"):].strip()
    return pred


def build_prompt_parts(tokenizer, document):
    """Split the chat-formatted prompt into a document part (cached) and a suffix added after each question."""
    separator = "#" * 50
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": INSTRUCTION + document + separator}],
        add_generation_prompt=True,
        tokenize=False,
    )
    context, question_suffix = text.split(separator)
    return context, question_suffix


def quantize_per_token(x, bits):
    """Asymmetric round-to-nearest quantization with one (scale, min) per token per head, then dequantize."""
    x_float = x.float()
    x_min = x_float.amin(dim=-1, keepdim=True)
    x_max = x_float.amax(dim=-1, keepdim=True)
    levels = 2**bits - 1
    scale = (x_max - x_min).clamp(min=1e-8) / levels
    q = ((x_float - x_min) / scale).round().clamp(0, levels)
    return (q * scale + x_min).to(x.dtype)


def cache_lengths(cache):
    return [layer.keys.shape[-2] for layer in cache.layers]


def trim_cache(cache, lengths):
    for layer, n in zip(cache.layers, lengths):
        layer.keys = layer.keys[:, :, :n]
        layer.values = layer.values[:, :, :n]


@torch.no_grad()
def answer_question(model, tokenizer, cache, question_ids, gold_ids, context_length, max_new_tokens):
    """Greedy-decode an answer and compute the gold answer's log-prob, leaving the cache unchanged."""
    device = model.device
    lengths = cache_lengths(cache)

    # Teacher-forced log-prob of the gold answer.
    ids = torch.cat([question_ids, gold_ids], dim=1).to(device)
    position_ids = torch.arange(context_length, context_length + ids.shape[1], device=device).unsqueeze(0)
    logits = model(input_ids=ids, past_key_values=cache, position_ids=position_ids).logits[0].float()
    n_q = question_ids.shape[1]
    log_probs = torch.log_softmax(logits[n_q - 1 : -1], dim=-1)
    gold_logprob = log_probs.gather(1, gold_ids[0].to(device).unsqueeze(1)).sum().item()
    trim_cache(cache, lengths)

    # Greedy generation.
    position_ids = torch.arange(context_length, context_length + n_q, device=device).unsqueeze(0)
    out = model(input_ids=question_ids.to(device), past_key_values=cache, position_ids=position_ids, logits_to_keep=1)
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    generated = [out.logits[0, -1].argmax()]
    position_ids = position_ids[:, -1:] + 1
    for i in range(max_new_tokens - 1):
        if generated[-1].item() in eos:
            break
        out = model(input_ids=generated[-1].view(1, 1), past_key_values=cache, position_ids=position_ids + i)
        generated.append(out.logits[0, -1].argmax())
    trim_cache(cache, lengths)
    pred = tokenizer.decode(torch.stack(generated), skip_special_tokens=True).strip()
    return pred, gold_logprob


def prefill(model, context_ids, press=None):
    cache = DynamicCache()
    if press is None:
        model.model(input_ids=context_ids.to(model.device), past_key_values=cache)
    else:
        with press(model):
            model.model(input_ids=context_ids.to(model.device), past_key_values=cache)
    return cache


def run_questions(model, tokenizer, cache, item, question_suffix, context_length, max_new_tokens, cot):
    # gold_logprob is always the direct-answer log-prob (QUESTION_TEMPLATE), even in CoT mode.
    template = COT_QUESTION_TEMPLATE if cot else QUESTION_TEMPLATE
    result = {}
    for key in ("hop1", "hop2", "two_hop"):
        gold_ids = tokenizer.encode(item[key]["a"], return_tensors="pt", add_special_tokens=False)
        direct_ids = tokenizer.encode(QUESTION_TEMPLATE.format(q=item[key]["q"]) + question_suffix,
                                      return_tensors="pt", add_special_tokens=False)
        q_ids = tokenizer.encode(template.format(q=item[key]["q"]) + question_suffix,
                                 return_tensors="pt", add_special_tokens=False)
        pred, lp = answer_question(model, tokenizer, cache, q_ids, gold_ids, context_length, max_new_tokens)
        if cot:
            _, lp = answer_question(model, tokenizer, cache, direct_ids, gold_ids, context_length, 1)
        answer = extract_answer(pred, cot)
        result[key] = {"pred": pred, "correct": item[key]["a"].lower() in answer.lower(), "gold_logprob": lp}
    return result


def kv_entries(cache):
    return sum(layer.keys.shape[-2] * layer.keys.shape[1] for layer in cache.layers)


@torch.inference_mode()
def evaluate_item(model, tokenizer, item, conditions, max_new_tokens, cot):
    from kvpress import RandomPress

    rows = []
    context, question_suffix = build_prompt_parts(tokenizer, item["context"])
    context_ids = tokenizer.encode(context, return_tensors="pt", add_special_tokens=False)
    n_ctx = context_ids.shape[1]

    full_cache = None
    for cond in conditions:
        t0 = time.time()
        stats = {}
        if cond == "nocontext":
            ctx, suffix = build_prompt_parts(tokenizer, "")
            ids = tokenizer.encode(ctx, return_tensors="pt", add_special_tokens=False)
            cache = prefill(model, ids)
            res = run_questions(model, tokenizer, cache, item, suffix, ids.shape[1], max_new_tokens, cot)
        elif cond.startswith("rand"):
            ratio = float(cond[4:])
            torch.manual_seed(item["id"])
            cache = prefill(model, context_ids, RandomPress(compression_ratio=ratio))
            full_entries = n_ctx * model.config.num_hidden_layers * model.config.num_key_value_heads
            stats = {"kv_fraction": kv_entries(cache) / full_entries}
            res = run_questions(model, tokenizer, cache, item, question_suffix, n_ctx, max_new_tokens, cot)
        else:
            if full_cache is None:
                full_cache = prefill(model, context_ids)
            if cond == "full":
                res = run_questions(model, tokenizer, full_cache, item, question_suffix, n_ctx, max_new_tokens, cot)
            elif cond.startswith("int"):
                bits = int(cond[3:])
                originals = [(layer.keys, layer.values) for layer in full_cache.layers]
                for layer in full_cache.layers:
                    layer.keys = quantize_per_token(layer.keys, bits)
                    layer.values = quantize_per_token(layer.values, bits)
                head_dim = full_cache.layers[0].keys.shape[-1]
                # Each (token, head) group also stores an fp16 scale and min.
                stats = {"bits_per_value": bits + 32 / head_dim}
                res = run_questions(model, tokenizer, full_cache, item, question_suffix, n_ctx, max_new_tokens, cot)
                for layer, (k, v) in zip(full_cache.layers, originals):
                    layer.keys, layer.values = k, v
            else:
                raise ValueError(f"unknown condition {cond}")
        rows.append({"id": item["id"], "config": item["config"], "condition": cond, "n_ctx_tokens": n_ctx,
                     **stats, **res, "seconds": round(time.time() - t0, 2)})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/two_hop.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--conditions", nargs="+", default=["full", "nocontext"])
    parser.add_argument("--n-per-config", type=int, default=None, help="use only the first k items of each config")
    parser.add_argument("--max-new-tokens", type=int, default=16, help="use ~160 with --cot")
    parser.add_argument("--cot", action="store_true", help="let the model reason before a final 'Answer:' line")
    parser.add_argument("--out", default="results/results.jsonl")
    args = parser.parse_args()

    items = [json.loads(line) for line in open(args.data)]
    if args.n_per_config is not None:
        seen = collections.Counter()
        kept = []
        for it in items:
            key = json.dumps(it["config"], sort_keys=True)
            if seen[key] < args.n_per_config:
                seen[key] += 1
                kept.append(it)
        items = kept

    # Resume: skip (id, condition) pairs that are already in the output file.
    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            row = json.loads(line)
            done.add((row["id"], row["condition"]))

    if torch.cuda.is_available():
        device, dtype = "cuda", torch.bfloat16
    elif torch.backends.mps.is_available():
        device, dtype = "mps", torch.float16
    else:
        device, dtype = "cpu", torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, attn_implementation="sdpa").to(device)
    model.eval()
    print(f"model={args.model} device={device} items={len(items)} conditions={args.conditions}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    start = time.time()
    with open(args.out, "a") as f:
        for i, item in enumerate(items):
            todo = [c for c in args.conditions if (item["id"], c) not in done]
            if not todo:
                continue
            for row in evaluate_item(model, tokenizer, item, todo, args.max_new_tokens, args.cot):
                f.write(json.dumps(row) + "\n")
            f.flush()
            if (i + 1) % 20 == 0:
                print(f"{i + 1}/{len(items)} items, {time.time() - start:.0f}s", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
