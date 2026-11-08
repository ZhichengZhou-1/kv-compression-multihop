"""Summarize eval_kv.py results.

For each condition (and optionally each config field) reports:
    hop1 / hop2 / two_hop accuracy
    indep        expected two-hop accuracy if the two retrievals failed independently (= acc_hop1 * acc_hop2)
    both_single  fraction of items where hop1 and hop2 are both correct
    comp_fail    P(two_hop wrong | hop1 and hop2 both correct), the per-item composition-failure rate
    logprob      mean gold-answer log-prob for each question type
    memory       fraction of full-precision KV memory kept

Usage:
    python summarize.py results/baseline.jsonl
    python summarize.py results/*.jsonl --by n_distractors

Restoration conditions (<base>+<set>) are listed right after their base, each followed by its _rnd control.
"""

import argparse
import collections
import json
import re

KEYS = ("hop1", "hop2", "two_hop")
RESTORE_ORDER = ("fact1", "fact2", "both", "after", "bothafter")


def condition_order(cond):
    """full, then rand by ratio, then quantization by bits (int before kivi, both K/V before K-only before
    V-only), then nocontext; restoration variants sort right after their base, each target before its control."""
    base, _, restore = cond.partition("+")
    if restore:
        target = restore.removesuffix("_rnd")
        rank = RESTORE_ORDER.index(target) if target in RESTORE_ORDER else len(RESTORE_ORDER)
        restore_key = (1, rank, restore.endswith("_rnd"))
    else:
        restore_key = (0, 0, False)
    if base == "full":
        base_key = (0, 0, 0, 0)
    elif base == "nocontext":
        base_key = (3, 0, 0, 0)
    elif base.startswith("rand"):
        base_key = (1, float(base[4:]), 0, 0)
    elif m := re.fullmatch(r"(int|kivi)(\d+)([kv]?)", base):
        base_key = (2, -int(m.group(2)), m.group(1) == "kivi", {"": 0, "k": 1, "v": 2}[m.group(3)])
    else:
        base_key = (4, 0, 0, 0)
    return base_key + restore_key + (cond,)


def memory_fraction(row):
    if "kv_fraction" in row:
        return row["kv_fraction"]
    if "bits_per_value" in row:
        return row["bits_per_value"] / 16
    return 1.0


def summarize(rows):
    n = len(rows)
    acc = {k: sum(r[k]["correct"] for r in rows) / n for k in KEYS}
    both = [r for r in rows if r["hop1"]["correct"] and r["hop2"]["correct"]]
    comp_fail = sum(not r["two_hop"]["correct"] for r in both) / len(both) if both else float("nan")
    return {
        "n": n,
        **{f"acc_{k}": acc[k] for k in KEYS},
        "indep": acc["hop1"] * acc["hop2"],
        "both_single": len(both) / n,
        "comp_fail": comp_fail,
        **{f"lp_{k}": sum(r[k]["gold_logprob"] for r in rows) / n for k in KEYS},
        "memory": sum(memory_fraction(r) for r in rows) / n,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+")
    parser.add_argument("--by", default=None, help="also split by a config field, e.g. n_distractors, pos_pair, order")
    args = parser.parse_args()

    rows = [json.loads(line) for path in args.files for line in open(path)]
    groups = collections.defaultdict(list)
    for r in rows:
        sub = json.dumps(r["config"][args.by]) if args.by else ""
        groups[(r["condition"], sub)].append(r)

    header = f"{'condition':<22} {args.by or '':>14} {'n':>5} {'mem':>5} {'hop1':>6} {'hop2':>6} {'2hop':>6} " \
             f"{'indep':>6} {'both1':>6} {'cfail':>6} {'lp_h1':>7} {'lp_h2':>7} {'lp_2h':>7}"
    print(header)
    print("-" * len(header))
    for (cond, sub) in sorted(groups, key=lambda g: (condition_order(g[0]), g[1])):
        s = summarize(groups[(cond, sub)])
        print(f"{cond:<22} {sub:>14} {s['n']:>5} {s['memory']:>5.2f} {s['acc_hop1']:>6.3f} {s['acc_hop2']:>6.3f} "
              f"{s['acc_two_hop']:>6.3f} {s['indep']:>6.3f} {s['both_single']:>6.3f} {s['comp_fail']:>6.3f} "
              f"{s['lp_hop1']:>7.2f} {s['lp_hop2']:>7.2f} {s['lp_two_hop']:>7.2f}")


if __name__ == "__main__":
    main()
