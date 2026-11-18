"""Evaluate paired single-hop / two-hop questions under KV cache compression.

The document is prefilled once per condition; the cache is then compressed (or not) and all three
questions (hop1, hop2, two_hop) are answered from the *same* cache. After each question the cache is
trimmed back to the document, so questions never see each other.

Conditions:
    full         uncompressed cache
    nocontext    questions asked with an empty document (checks answers need the context)
    rand<r>      kvpress RandomPress, evicting a fraction r of document KV entries (e.g. rand0.5)
    snap<r>      kvpress SnapKVPress (scores from the attention of the last 64 context tokens)
    stream<r>    kvpress StreamingLLMPress (keeps 4 sink tokens plus the most recent window)
    h2o<r>       kvpress ObservedAttentionPress (H2O-style average received attention; prefills with eager attention)
    expatt<r>    kvpress ExpectedAttentionPress
                 For the scored presses the row also records, under "kept", the fraction of each fact, of the
                 tokens after each fact, of the prompt prefix and of the whole document that survives eviction.
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
    <base>+afterearly  only the tokens after the fact that comes first in the document (they cannot see the later fact)
    <base>+afterlate   only the tokens after the fact that comes second (the only 'after' tokens that see both facts)
    <base>+bothearly   both facts plus afterearly;  <base>+bothlate: both facts plus afterlate
    <base>+prefix      every position before the document (chat template, system prompt, instruction, attention sinks)
    <base>+none        nothing restored: the built-in eviction alone, the baseline for the other restore sets
    <base>+all         every context position (sanity check: should reproduce 'full' exactly; no _rnd control)
    <base>+<set>_rnd   control: the same number of random document positions, excluding the facts and <set>;
                       for bothafter/bothearly/bothlate the facts stay restored and only the 'after' part is
                       replaced by random positions
    Restored entries are kept on top of the compression budget. rand<r>+<set> uses a built-in random eviction
    that mirrors RandomPress and shares its random draw across all restore sets of an item.

Extra flags:
    --paraphrases      also ask each single-hop question in two more wordings (keys hop1_p1, hop1_p2, hop2_p1, hop2_p2)
    --question-order   order in which the three questions are asked, e.g. two_hop:hop2:hop1 (default hop1:hop2:two_hop)
    --attn-probe       record, for hop2 and two_hop, the attention the question tokens pay to each fact and to the
                       restored positions, and the norm of the cached keys relative to the full cache (under "probe")

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
PRESS_RE = re.compile(r"^(snap|stream|h2o|expatt)([\d.]+)$")
RESTORE_SETS = ("fact1", "fact2", "both", "after", "bothafter", "afterearly", "afterlate", "bothearly", "bothlate",
                "prefix", "none", "all")
NO_CONTROL_SETS = ("none", "all")
QUESTION_KEYS = ("hop1", "hop2", "two_hop")
# Extra wordings of the single-hop questions, filled from item["entities"]; the gold answer is unchanged.
PARAPHRASES = {
    "hop1_p1": ("hop1", "Who is the father of {a}?"),
    "hop1_p2": ("hop1", "What is the name of {a}'s father?"),
    "hop2_p1": ("hop2", "Which company does {b} work at?"),
    "hop2_p2": ("hop2", "What is the name of {b}'s employer?"),
}


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

    spans = {
        "doc": tokens_in(0, len(document)),
        "fact1": tokens_in(*item["fact1_span"]),
        "fact2": tokens_in(*item["fact2_span"]),
    }
    spans["prefix"] = list(range(spans["doc"][0]))
    doc_end = spans["doc"][-1] + 1
    facts = set(spans["fact1"]) | set(spans["fact2"])
    early, late = sorted(("fact1", "fact2"), key=lambda k: spans[k][0])
    for name, key in (("after_early", early), ("after_late", late)):
        last, size = spans[key][-1], len(spans[key])
        spans[name] = sorted(set(range(last + 1, min(last + 1 + size, doc_end))) - facts)
    return spans


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
    if target == "none":
        return []
    if target in ("afterearly", "afterlate", "bothearly", "bothlate", "prefix"):
        chosen = set(spans["prefix"] if target == "prefix" else
                     spans["after_early"] if target in ("afterearly", "bothearly") else spans["after_late"])
        with_facts = target.startswith("both")
        if control:
            pool = sorted(set(spans["doc"]) - facts - chosen)
            chosen = set(random.Random(seed).sample(pool, len(chosen)))
        return sorted(chosen | facts) if with_facts else sorted(chosen)
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
        idx = scores.topk(min(n, n_kept + len(keep)), dim=-1).indices.sort(dim=-1).values
        idx = idx.to(layer.keys.device)[None, :, :, None]
        layer.keys = layer.keys.gather(2, idx.expand(-1, -1, -1, layer.keys.shape[-1]))
        layer.values = layer.values.gather(2, idx.expand(-1, -1, -1, layer.values.shape[-1]))


def make_press(name, ratio):
    """A kvpress scorer press whose kept indices per layer are recorded in press.kept_indices."""
    from kvpress import ExpectedAttentionPress, ObservedAttentionPress, SnapKVPress, StreamingLLMPress
    from kvpress.utils import compute_n_kept

    press = {"snap": SnapKVPress, "stream": StreamingLLMPress, "h2o": ObservedAttentionPress,
             "expatt": ExpectedAttentionPress}[name](compression_ratio=ratio)
    press.kept_indices = {}

    def compress(module, hidden_states, keys, values, attentions, kwargs):
        # Same as ScorerPress.compress, but remembers which positions each head keeps.
        scores = press.score(module, hidden_states, keys, values, attentions, kwargs)
        n_kept = compute_n_kept(keys.shape[2], press.compression_ratio)
        indices = scores.topk(n_kept, dim=-1).indices.sort(dim=-1).values
        press.kept_indices[module.layer_idx] = indices[0].cpu()
        gather_idx = indices.unsqueeze(-1).expand(-1, -1, -1, keys.shape[-1])
        return keys.gather(2, gather_idx).contiguous(), values.gather(2, gather_idx).contiguous()

    press.compress = compress
    return press


def kept_fractions(kept_indices, n_ctx, spans):
    """Fraction of each position set that survives eviction, averaged over layers and KV heads."""
    masks = []
    for idx in kept_indices.values():
        m = torch.zeros(idx.shape[0], n_ctx, dtype=torch.bool)
        m.scatter_(1, idx, True)
        masks.append(m)
    mask = torch.stack(masks)  # (layers, heads, n_ctx)
    sets = {k: spans[k] for k in ("fact1", "fact2", "after_early", "after_late", "prefix", "doc")}
    return {k: round(mask[:, :, v].float().mean().item(), 4) if v else None for k, v in sets.items()}


@torch.no_grad()
def attention_probe(model, cache, question_ids, context_length, position_sets, originals=None):
    """Attention the question tokens pay to each position set (summed over the set, averaged over heads; one value
    per layer for the last question token and for the mean over question tokens). With `originals` (the
    full-precision K/V), also the mean key norm of each set relative to the full cache, per layer."""
    lengths = cache_lengths(cache)
    device = model.device
    model.set_attn_implementation("eager")
    try:
        position_ids = torch.arange(context_length, context_length + question_ids.shape[1], device=device)[None]
        out = model(input_ids=question_ids.to(device), past_key_values=cache, position_ids=position_ids,
                    output_attentions=True, logits_to_keep=1)
    finally:
        model.set_attn_implementation("sdpa")
        trim_cache(cache, lengths)
    probe = {}
    for name, positions in position_sets.items():
        if not positions:
            continue
        idx = torch.tensor(positions, device=device)
        last = [a[0, :, -1, idx].float().sum(-1).mean().item() for a in out.attentions]
        mean = [a[0, :, :, idx].float().sum(-1).mean().item() for a in out.attentions]
        entry = {"attn_last": [round(x, 5) for x in last], "attn_mean": [round(x, 5) for x in mean]}
        if originals is not None:
            ratios = []
            for layer, (k_full, _) in zip(cache.layers, originals):
                n_cached = layer.keys[0, :, idx].float().norm(dim=-1).mean()
                n_full = k_full[0, :, idx].float().norm(dim=-1).mean()
                ratios.append(round((n_cached / n_full).item(), 4))
            entry["key_norm_ratio"] = ratios
        probe[name] = entry
    return probe


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


def run_questions(model, tokenizer, cache, item, question_suffix, context_length, max_new_tokens, cot,
                  order=QUESTION_KEYS, paraphrases=False, probe_sets=None, originals=None):
    # gold_logprob is always the direct-answer log-prob (QUESTION_TEMPLATE), even in CoT mode; paraphrased
    # questions only get the log-prob under their own (CoT) prompt.
    template = COT_QUESTION_TEMPLATE if cot else QUESTION_TEMPLATE
    questions = [(key, item[key]["q"], item[key]["a"], True) for key in order]
    if paraphrases:
        questions += [(key, q.format(**item["entities"]), item[src]["a"], False)
                      for key, (src, q) in PARAPHRASES.items()]
    result = {}
    for key, question, gold, direct_lp in questions:
        gold_ids = tokenizer.encode(gold, return_tensors="pt", add_special_tokens=False)
        q_ids = tokenizer.encode(template.format(q=question) + question_suffix,
                                 return_tensors="pt", add_special_tokens=False)
        pred, lp = answer_question(model, tokenizer, cache, q_ids, gold_ids, context_length, max_new_tokens)
        if cot and direct_lp:
            direct_ids = tokenizer.encode(QUESTION_TEMPLATE.format(q=question) + question_suffix,
                                          return_tensors="pt", add_special_tokens=False)
            _, lp = answer_question(model, tokenizer, cache, direct_ids, gold_ids, context_length, 1)
        answer = extract_answer(pred, cot)
        result[key] = {"pred": pred, "correct": gold.lower() in answer.lower(), "gold_logprob": lp}
        if probe_sets is not None and key in ("hop2", "two_hop"):
            result[key]["probe"] = attention_probe(model, cache, q_ids, context_length, probe_sets, originals)
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
    if target in NO_CONTROL_SETS and control:
        raise ValueError(f"{cond}: the '{target}' restore set has no random control")
    if not (base.startswith("rand") or parse_quant(base)):
        raise ValueError(f"restoration needs a rand<r>, int<b> or kivi<b> base, got {cond}")
    return base, target, control


@torch.inference_mode()
def evaluate_item(model, tokenizer, item, conditions, max_new_tokens, cot, order=QUESTION_KEYS, paraphrases=False,
                  attn_probe=False):
    rows = []
    context, question_suffix = build_prompt_parts(tokenizer, item["context"])
    context_ids = tokenizer.encode(context, return_tensors="pt", add_special_tokens=False)
    n_ctx = context_ids.shape[1]
    full_entries = n_ctx * model.config.num_hidden_layers * model.config.num_key_value_heads
    spans = document_token_spans(tokenizer, context, item["context"], item)
    ask = dict(order=order, paraphrases=paraphrases)

    def probe_sets(restored=()):
        # Positions only line up with the full cache when nothing is evicted, so the probe is for full/quant only.
        if not attn_probe:
            return None
        sets = {"fact1": spans["fact1"], "fact2": spans["fact2"], "sink": spans["prefix"][:4]}
        extra = sorted(set(restored) - set(spans["fact1"]) - set(spans["fact2"]))
        if extra:
            sets["restored_other"] = extra
        return sets

    full_cache = None
    for cond in conditions:
        t0 = time.time()
        stats = {}
        base, target, control = parse_condition(cond)
        press_match = PRESS_RE.match(base)
        if base == "nocontext":
            ctx, suffix = build_prompt_parts(tokenizer, "")
            ids = tokenizer.encode(ctx, return_tensors="pt", add_special_tokens=False)
            cache = prefill(model, ids)
            res = run_questions(model, tokenizer, cache, item, suffix, ids.shape[1], max_new_tokens, cot, **ask)
        elif base.startswith("rand") and target is None:
            from kvpress import RandomPress

            ratio = float(base[4:])
            torch.manual_seed(item["id"])
            cache = prefill(model, context_ids, RandomPress(compression_ratio=ratio))
            stats = {"kv_fraction": kv_entries(cache) / full_entries}
            res = run_questions(model, tokenizer, cache, item, question_suffix, n_ctx, max_new_tokens, cot, **ask)
        elif press_match:
            name, ratio = press_match.group(1), float(press_match.group(2))
            press = make_press(name, ratio)
            if name == "h2o":  # needs the attention weights during prefill
                model.set_attn_implementation("eager")
            try:
                cache = prefill(model, context_ids, press)
            finally:
                model.set_attn_implementation("sdpa")
            stats = {"kv_fraction": kv_entries(cache) / full_entries,
                     "kept": kept_fractions(press.kept_indices, n_ctx, spans)}
            res = run_questions(model, tokenizer, cache, item, question_suffix, n_ctx, max_new_tokens, cot, **ask)
            del cache
        elif base == "full" and target is None:
            if full_cache is None:
                full_cache = prefill(model, context_ids)
            res = run_questions(model, tokenizer, full_cache, item, question_suffix, n_ctx, max_new_tokens, cot,
                                probe_sets=probe_sets(), **ask)
        elif base.startswith("rand") or parse_quant(base):
            if full_cache is None:
                full_cache = prefill(model, context_ids)
            keep = []
            if target == "all":
                keep = list(range(n_ctx))
                stats["n_restored"] = n_ctx
            elif target is not None:
                keep = restore_positions(spans, target, control, seed=f"{item['id']}-{target}")
                stats["n_restored"] = len(keep)
            originals = [(layer.keys, layer.values) for layer in full_cache.layers]

            sets = None
            if base.startswith("rand"):
                evict_random(full_cache, float(base[4:]), keep, seed=item["id"])
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
                sets = probe_sets(keep)

            res = run_questions(model, tokenizer, full_cache, item, question_suffix, n_ctx, max_new_tokens, cot,
                                probe_sets=sets, originals=originals if sets else None, **ask)
            for layer, (k, v) in zip(full_cache.layers, originals):
                layer.keys, layer.values = k, v
        else:
            raise ValueError(f"unknown condition {cond}")
        row = {"id": item["id"], "config": item["config"], "condition": cond, "n_ctx_tokens": n_ctx, **stats, **res,
               "seconds": round(time.time() - t0, 2)}
        if tuple(order) != QUESTION_KEYS:
            row["question_order"] = list(order)
        rows.append(row)
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
    parser.add_argument("--paraphrases", action="store_true", help="also ask two more wordings of each single-hop question")
    parser.add_argument("--question-order", default=",".join(QUESTION_KEYS),
                        help="order of the three questions, e.g. two_hop:hop2:hop1 (commas also work)")
    parser.add_argument("--attn-probe", action="store_true",
                        help="record attention to the facts and key-norm ratios (full and quantization conditions)")
    args = parser.parse_args()
    order = tuple(re.split(r"[,:]", args.question_order))  # ":" also works inside sbatch --export
    if sorted(order) != sorted(QUESTION_KEYS):
        parser.error(f"--question-order must be a permutation of {','.join(QUESTION_KEYS)}")
    for cond in args.conditions:  # fail on a typo before loading the model
        base, _, _ = parse_condition(cond)
        if not (base in ("full", "nocontext") or parse_quant(base) or re.fullmatch(r"rand[\d.]+", base)
                or PRESS_RE.match(base)):
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
            for row in evaluate_item(model, tokenizer, item, todo, args.max_new_tokens, args.cot, order=order,
                                     paraphrases=args.paraphrases, attn_probe=args.attn_probe):
                f.write(json.dumps(row) + "\n")
            f.flush()
            if (i + 1) % 20 == 0:
                print(f"{i + 1}/{len(items)} items, {time.time() - start:.0f}s", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
