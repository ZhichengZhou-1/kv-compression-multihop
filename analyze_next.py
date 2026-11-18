"""Analyses for the follow-up experiments (paraphrases, scored presses, attention probe, question order).

Usage:
    python analyze_next.py paraphrase results/para_*.jsonl
    python analyze_next.py kept results/press_*.jsonl
    python analyze_next.py probe results/probe_kivi2.jsonl
    python analyze_next.py order results/order_check.jsonl results/sweep_base.jsonl results/sweep_rand.jsonl ...
    python analyze_next.py newcf results/sink_*.jsonl --ref results/sweep_base.jsonl

paraphrase: for each condition, how often a single-hop fact found with the original wording is also found with
    both other wordings, and the composition failure rate (CFR) split by whether the single-hop answers are robust
    (all three wordings of both hops correct) or wording-dependent.
kept: fraction of each fact / after-tokens / prefix that each scored press keeps, next to its accuracies.
probe: attention (averaged over layers) that the hop2 / two_hop question pays to each fact and to other restored
    positions, and the key-norm ratio of the cached facts relative to the full cache.
order: share of (item, condition, question) predictions that are identical to a reference run with the default
    question order.
newcf: J, CFR and new composition failures (items fully correct with the full cache whose single-hop answers
    survive compression but whose two-hop answer is now wrong), using --ref for the full-cache run.
"""

import argparse
import collections
import json
import statistics

from summarize import condition_order

HOPS = ("hop1", "hop2", "two_hop")


def load(paths):
    groups = collections.defaultdict(dict)
    for path in paths:
        for line in open(path):
            row = json.loads(line)
            groups[row["condition"]][row["id"]] = row
    return groups


def ordered(groups):
    return sorted(groups, key=condition_order)


def frac(k, n):
    return f"{k / n:.2f} ({k}/{n})" if n else "--"


def paraphrase(groups):
    print(f"{'condition':<22} {'n':>4} {'hop1 robust':>16} {'hop2 robust':>16} {'CFR all':>16} "
          f"{'CFR robust':>16} {'CFR not robust':>16}")
    for cond in ordered(groups):
        rows = list(groups[cond].values())
        if "hop1_p1" not in rows[0]:
            continue
        robust = {}
        for hop in ("hop1", "hop2"):
            found = [r for r in rows if r[hop]["correct"]]
            ok = [r for r in found if r[f"{hop}_p1"]["correct"] and r[f"{hop}_p2"]["correct"]]
            robust[hop] = (len(ok), len(found))
        both = [r for r in rows if r["hop1"]["correct"] and r["hop2"]["correct"]]
        is_robust = lambda r: all(r[k]["correct"] for k in ("hop1_p1", "hop1_p2", "hop2_p1", "hop2_p2"))
        strong = [r for r in both if is_robust(r)]
        weak = [r for r in both if not is_robust(r)]
        fails = lambda rs: sum(not r["two_hop"]["correct"] for r in rs)
        print(f"{cond:<22} {len(rows):>4} {frac(*robust['hop1']):>16} {frac(*robust['hop2']):>16} "
              f"{frac(fails(both), len(both)):>16} {frac(fails(strong), len(strong)):>16} "
              f"{frac(fails(weak), len(weak)):>16}")


def kept(groups):
    sets = ("fact1", "fact2", "after_early", "after_late", "prefix", "doc")
    print(f"{'condition':<14} {'n':>4} " + " ".join(f"{s:>11}" for s in sets) + f" {'hop1':>6} {'hop2':>6} {'2hop':>6}")
    for cond in ordered(groups):
        rows = list(groups[cond].values())
        if "kept" not in rows[0]:
            continue
        vals = [statistics.mean(r["kept"][s] for r in rows if r["kept"][s] is not None) for s in sets]
        acc = [sum(r[h]["correct"] for r in rows) / len(rows) for h in HOPS]
        print(f"{cond:<14} {len(rows):>4} " + " ".join(f"{v:>11.3f}" for v in vals) + " "
              + " ".join(f"{a:>6.3f}" for a in acc))


def probe(groups):
    for question in ("hop2", "two_hop"):
        print(f"\n{question}: attention from the last question token (mean over question tokens), layer average")
        print(f"{'condition':<20} {'n':>4} {'fact1':>15} {'fact2':>15} {'other restored':>15} {'sink':>15} "
              f"{'key norm fact1':>15} {'key norm fact2':>15} {'acc':>6}")
        for cond in ordered(groups):
            rows = [r for r in groups[cond].values() if "probe" in r[question]]
            if not rows:
                continue

            def avg(name, field):
                vals = [statistics.mean(r[question]["probe"][name][field]) for r in rows
                        if name in r[question]["probe"] and field in r[question]["probe"][name]]
                return statistics.mean(vals) if vals else None

            def cell(name):
                last, mean = avg(name, "attn_last"), avg(name, "attn_mean")
                return f"{last:.4f} ({mean:.4f})" if last is not None else "--"

            def norm(name):
                v = avg(name, "key_norm_ratio")
                return f"{v:.3f}" if v is not None else "--"

            acc = sum(r[question]["correct"] for r in rows) / len(rows)
            print(f"{cond:<20} {len(rows):>4} {cell('fact1'):>15} {cell('fact2'):>15} {cell('restored_other'):>15} "
                  f"{cell('sink'):>15} {norm('fact1'):>15} {norm('fact2'):>15} {acc:>6.3f}")


def order(paths):
    test, refs = load(paths[:1]), load(paths[1:])
    for cond in ordered(test):
        same = total = changed_correct = 0
        for item_id, row in test[cond].items():
            ref = refs.get(cond, {}).get(item_id)
            if ref is None:
                continue
            for h in HOPS:
                total += 1
                same += row[h]["pred"] == ref[h]["pred"]
                changed_correct += row[h]["correct"] != ref[h]["correct"]
        print(f"{cond:<14} identical predictions {frac(same, total)}, correctness changed in {changed_correct}")


def newcf(groups, ref_paths):
    ref = load(ref_paths)["full"]
    base = {i for i, r in ref.items() if all(r[h]["correct"] for h in HOPS)}
    print(f"{'condition':<22} {'n':>4} {'2hop':>6} {'J':>6} {'CFR':>16} {'New CF':>16}")
    for cond in ordered(groups):
        rows = groups[cond]
        both = [i for i, r in rows.items() if r["hop1"]["correct"] and r["hop2"]["correct"]]
        still = [i for i in both if i in base]
        two = sum(r["two_hop"]["correct"] for r in rows.values()) / len(rows)
        print(f"{cond:<22} {len(rows):>4} {two:>6.3f} {len(both) / len(rows):>6.3f} "
              f"{frac(sum(not rows[i]['two_hop']['correct'] for i in both), len(both)):>16} "
              f"{frac(sum(not rows[i]['two_hop']['correct'] for i in still), len(still)):>16}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("analysis", choices=["paraphrase", "kept", "probe", "order", "newcf"])
    parser.add_argument("files", nargs="+")
    parser.add_argument("--ref", nargs="+", default=["results/sweep_base.jsonl"], help="full-cache run for newcf")
    args = parser.parse_args()
    if args.analysis == "order":
        order(args.files)
    elif args.analysis == "newcf":
        newcf(load(args.files), args.ref)
    else:
        globals()[args.analysis](load(args.files))


if __name__ == "__main__":
    main()
