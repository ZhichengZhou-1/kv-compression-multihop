"""Evaluate paired single-hop / two-hop questions under KV cache compression.

The document is prefilled once per condition; the cache is then compressed (or not) and all three
questions (hop1, hop2, two_hop) are answered from the *same* cache. After each question the cache is
trimmed back to the document, so questions never see each other.

Conditions:
    full         uncompressed cache
    nocontext    questions asked with an empty document (checks answers need the context)
    rand<r>      kvpress RandomPress, evicting a fraction r of document KV entries (e.g. rand0.5)
    int<b>       per-token round-to-nearest fake quantization of document K/V to b bits (e.g. int4)
    int<b>k      per-token quantization of K only (V stays fp16); int<b>v quantizes V only
    kivi<b>      KIVI-style: K quantized per channel in groups of 32 tokens, V per token
    kivi<b>k     per-channel K only (V stays fp16); kivi<b>v is the same as int<b>v

Restoration (appended to a rand<r>, int*/kivi* condition with '+'):
    <base>+fact1       keep fact 1's tokens at full fidelity (not evicted / not quantized)
    <base>+fact2       same for fact 2
    <base>+both        both facts
    <base>+after       for each fact, the same number of tokens right after it
    <base>+bothafter   both facts plus the 'after' tokens
    <base>+<set>_rnd   control: the same number of random document positions, excluding the facts and <set>;
                       for bothafter the facts stay restored and only the 'after' part is replaced by random positions
    Restored entries are kept on top of the compression budget. rand<r>+<set> uses a built-in random eviction
    that mirrors RandomPress and shares its random draw across all restore sets of an item.

Usage:
    python eval_kv.py --data data/two_hop.jsonl --n-per-config 5 --conditions full nocontext
    python eval_kv.py --conditions full nocontext rand0.25 rand0.5 rand0.75 int8 int4 int3 int2
    python eval_kv.py --conditions int4 int4k int4v kivi4 kivi4k kivi2
    python eval_kv.py --conditions int4 int4+fact1 int4+fact1_rnd int4+both int4+both_rnd int4+after int4+after_rnd
"""

import argparse
import collections
import json
import os
import random
import re
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


KIVI_GROUP_SIZE = 32  # tokens per (scale, min) group for per-channel key quantization
QUANT_RE = re.compile(r"^(int|kivi)(\d+)([kv]?)$")
RESTORE_SETS = ("fact1", "fact2", "both", "after", "bothafter")


def _fake_quantize(x, bits, dim):
    """Asymmetric round-to-nearest quantization with one (scale, min) per slice along `dim`, then dequantize."""
    x_float = x.float()
    x_min = x_float.amin(dim=dim, keepdim=True)
    x_max = x_float.amax(dim=dim, keepdim=True)
    levels = 2**bits - 1
    scale = (x_max - x_min).clamp(min=1e-8) / levels
    q = ((x_float - x_min) / scale).round().clamp(0, levels)
    return (q * scale + x_min).to(x.dtype)


def quantize_per_token(x, bits):
    """One (scale, min) per token per head, shared across the head dimension."""
    return _fake_quantize(x, bits, dim=-1)


def quantize_per_channel(x, bits, group_size=KIVI_GROUP_SIZE):
    """KIVI-style: one (scale, min) per channel per head, shared across groups of `group_size` tokens."""
    return torch.cat([_fake_quantize(x[:, :, s:s + group_size], bits, dim=-2)
                      for s in range(0, x.shape[-2], group_size)], dim=-2)


def parse_quant(cond):
    """Map a quantization condition to (key_mode, value_mode, bits); modes are 'token', 'channel' or None.

    int<b>   per-token K and V          int<b>k / int<b>v    per-token K only / V only
    kivi<b>  per-channel K, per-token V  kivi<b>k / kivi<b>v  per-channel K only / per-token V only
    """
    m = QUANT_RE.match(cond)
    if m is None:
        return None
    scheme, bits, only = m.group(1), int(m.group(2)), m.group(3)
    key_mode = "channel" if scheme == "kivi" else "token"
    return (key_mode if only != "v" else None, "token" if only != "k" else None, bits)


def bits_per_value(mode, bits, head_dim):
    """Storage cost per K or V entry, including an fp16 scale and min per quantization group."""
    if mode is None:
        return 16.0
    return bits + 32 / (head_dim if mode == "token" else KIVI_GROUP_SIZE)


def quantize_tensor(x, mode, bits):
    if mode is None:
        return x
    return quantize_per_token(x, bits) if mode == "token" else quantize_per_channel(x, bits)


def document_token_spans(tokenizer, context, document, item):
    """Token indices (into the context ids) of the document and of each supporting fact."""
    doc_start = context.index(INSTRUCTION + document) + len(INSTRUCTION)
    enc = tokenizer(context, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]

    def tokens_in(start, end):
        return [i for i, (s, e) in enumerate(offsets) if s < doc_start + end and e > doc_start + start]

    return {
        "doc": tokens_in(0, len(document)),
        "fact1": tokens_in(*item["fact1_span"]),
        "fact2": tokens_in(*item["fact2_span"]),
    }


def restore_positions(spans, target, control, seed):
    """Context positions to keep at full fidelity.

    fact1 / fact2 / both: the supporting facts' tokens.
    after: for each fact, the same number of tokens immediately after it (excluding the facts themselves).
    bothafter: both facts plus the after tokens.
    control: the same number of positions drawn uniformly from the rest of the document, excluding both facts
             and the target set. For bothafter the facts are kept and only the after part is randomized, so
             bothafter vs bothafter_rnd isolates the after tokens given that the facts are present.
    """
    facts = set(spans["fact1"]) | set(spans["fact2"])
    doc_end = spans["doc"][-1] + 1
    if target == "fact1":
        chosen = set(spans["fact1"])
    elif target == "fact2":
        chosen = set(spans["fact2"])
    elif target == "both":
        chosen = facts
    elif target in ("after", "bothafter"):
        chosen = set()
        for key in ("fact1", "fact2"):
            last, size = spans[key][-1], len(spans[key])
            chosen |= set(range(last + 1, min(last + 1 + size, doc_end)))
        chosen -= facts
        if target == "bothafter":
            if not control:
                return sorted(chosen | facts)
            pool = sorted(set(spans["doc"]) - facts - chosen)
            return sorted(facts | set(random.Random(seed).sample(pool, len(chosen))))
    else:
        raise ValueError(f"unknown restore set {target}")
    if not control:
        return sorted(chosen)
    pool = sorted(set(spans["doc"]) - facts - chosen)
    return sorted(random.Random(seed).sample(pool, len(chosen)))


def evict_random(cache, ratio, keep, seed):
    """Random per-head eviction of a full cache, mirroring kvpress RandomPress (int(n * (1 - ratio)) entries
    kept per head), except that positions in `keep` are always kept on top of that budget.

    Compressing after a full prefill is equivalent to kvpress's prefill hooks: each layer's prefill attention
    sees that layer's full cache either way. The generator is seeded per item, so every restore set of an item
    shares the same random scores and the comparison between restore sets is paired.
    """
    n = cache.layers[0].keys.shape[-2]
    n_kept = int(n * (1 - ratio))
    keep = torch.tensor(keep, dtype=torch.long)
    gen = torch.Generator().manual_seed(seed)
    for layer in cache.layers:
        n_heads = layer.keys.shape[1]
        scores = torch.rand(n_heads, n, generator=gen)
        scores[:, keep] = 2.0  # above every random score, so always kept
        idx = scores.topk(n_kept + len(keep), dim=-1).indices.sort(dim=-1).values
        idx = idx.to(layer.keys.device)[None, :, :, None]
        layer.keys = layer.keys.gather(2, idx.expand(-1, -1, -1, layer.keys.shape[-1]))
        layer.values = layer.values.gather(2, idx.expand(-1, -1, -1, layer.values.shape[-1]))


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


def parse_condition(cond):
    """Split '<base>[+<restore set>[_rnd]]' into (base, restore set or None, is_random_control)."""
    base, _, restore = cond.partition("+")
    if not restore:
        return base, None, False
    control = restore.endswith("_rnd")
    target = restore[:-len("_rnd")] if control else restore
    if target not in RESTORE_SETS:
        raise ValueError(f"unknown restore set in {cond}; expected one of {RESTORE_SETS} (optionally + '_rnd')")
    if not (base.startswith("rand") or parse_quant(base)):
        raise ValueError(f"restoration needs a rand<r>, int<b> or kivi<b> base, got {cond}")
    return base, target, control


@torch.inference_mode()
def evaluate_item(model, tokenizer, item, conditions, max_new_tokens, cot):
    rows = []
    context, question_suffix = build_prompt_parts(tokenizer, item["context"])
    context_ids = tokenizer.encode(context, return_tensors="pt", add_special_tokens=False)
    n_ctx = context_ids.shape[1]
    spans = None

    full_cache = None
    for cond in conditions:
        t0 = time.time()
        stats = {}
        base, target, control = parse_condition(cond)
        if base == "nocontext":
            ctx, suffix = build_prompt_parts(tokenizer, "")
            ids = tokenizer.encode(ctx, return_tensors="pt", add_special_tokens=False)
            cache = prefill(model, ids)
            res = run_questions(model, tokenizer, cache, item, suffix, ids.shape[1], max_new_tokens, cot)
        elif base.startswith("rand") and target is None:
            from kvpress import RandomPress

            ratio = float(base[4:])
            torch.manual_seed(item["id"])
            cache = prefill(model, context_ids, RandomPress(compression_ratio=ratio))
            full_entries = n_ctx * model.config.num_hidden_layers * model.config.num_key_value_heads
            stats = {"kv_fraction": kv_entries(cache) / full_entries}
            res = run_questions(model, tokenizer, cache, item, question_suffix, n_ctx, max_new_tokens, cot)
        elif base == "full" and target is None:
            if full_cache is None:
                full_cache = prefill(model, context_ids)
            res = run_questions(model, tokenizer, full_cache, item, question_suffix, n_ctx, max_new_tokens, cot)
        elif base.startswith("rand") or parse_quant(base):
            if full_cache is None:
                full_cache = prefill(model, context_ids)
            keep = []
            if target is not None:
                if spans is None:
                    spans = document_token_spans(tokenizer, context, item["context"], item)
                keep = restore_positions(spans, target, control, seed=f"{item['id']}-{target}")
                stats["n_restored"] = len(keep)
            originals = [(layer.keys, layer.values) for layer in full_cache.layers]

            if base.startswith("rand"):
                evict_random(full_cache, float(base[4:]), keep, seed=item["id"])
                full_entries = n_ctx * model.config.num_hidden_layers * model.config.num_key_value_heads
                stats["kv_fraction"] = kv_entries(full_cache) / full_entries
            else:
                key_mode, value_mode, bits = parse_quant(base)
                for layer in full_cache.layers:
                    layer.keys = quantize_tensor(layer.keys, key_mode, bits)
                    layer.values = quantize_tensor(layer.values, value_mode, bits)
                if keep:
                    keep_idx = torch.tensor(keep, device=model.device)
                    for layer, (k, v) in zip(full_cache.layers, originals):
                        layer.keys[:, :, keep_idx] = k[:, :, keep_idx]
                        layer.values[:, :, keep_idx] = v[:, :, keep_idx]
                head_dim = full_cache.layers[0].keys.shape[-1]
                quantized = (bits_per_value(key_mode, bits, head_dim) + bits_per_value(value_mode, bits, head_dim)) / 2
                # Restored positions are stored in fp16.
                stats["bits_per_value"] = (quantized * (n_ctx - len(keep)) + 16 * len(keep)) / n_ctx

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
    parser.add_argument("--max-gpu-memory", default=None,
                        help="e.g. 13GiB: cap the GPU memory used for weights; layers that don't fit run from CPU RAM "
                             "(same model, slower). For GPUs too small to hold the model, such as 16 GB cards.")
    args = parser.parse_args()
    for cond in args.conditions:  # fail on a typo before loading the model
        base, _, _ = parse_condition(cond)
        if not (base in ("full", "nocontext") or parse_quant(base) or re.fullmatch(r"rand[\d.]+", base)):
            parser.error(f"unknown condition {cond}")

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
    if args.max_gpu_memory and device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=dtype, attn_implementation="sdpa",
            device_map="auto", max_memory={0: args.max_gpu_memory, "cpu": "64GiB"})
        offloaded = sorted({str(d) for d in model.hf_device_map.values()} - {"0", "cuda:0"})
        print(f"device_map: {sum(str(d) not in ('0', 'cuda:0') for d in model.hf_device_map.values())} "
              f"modules offloaded to {offloaded or 'nothing'}", flush=True)
    else:
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
