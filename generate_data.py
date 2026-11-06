"""Generate synthetic paired single-hop / two-hop questions over fictional entities.

Each item has a document made of short fact sentences:
    fact1: "<A>'s father is <B>."
    fact2: "<B> works at <C>."
plus distractor chains using the same two templates with other fictional people.

Questions (all answerable only from the document):
    hop1:    "Who is <A>'s father?"            -> B
    hop2:    "Where does <B> work?"            -> C
    two_hop: "Where does <A>'s father work?"   -> C

Usage:
    python generate_data.py --out data/two_hop.jsonl
    python generate_data.py --n-items 200 --n-distractors 20 50 100 --tokenizer Qwen/Qwen2.5-1.5B-Instruct
"""

import argparse
import itertools
import json
import os
import random

# Fathers draw from MALE_NAMES only; children draw from both lists.
MALE_NAMES = [
    "Liam", "Owen", "Elias", "Felix", "Jonah", "Caleb", "Silas", "Theo", "Hugo", "Isaac",
    "Miles", "Ezra", "Arlo", "Rowan", "Jasper", "Gideon", "Tobias", "Oscar", "Emmett", "Declan",
    "Nolan", "Ronan", "Victor", "Marcus", "Simon",
]
FEMALE_NAMES = [
    "Mira", "Nora", "Ada", "Iris", "Lena", "Ruth", "Vera", "Clara", "Maya", "Eva",
    "Hazel", "June", "Stella", "Alice", "Ivy", "Leah", "Greta", "Wren", "Lucy", "Freya",
    "Celia", "Daphne", "Esme", "Thea", "Paige",
]
SURNAME_SYLLABLES = [
    "vos", "hal", "quel", "pem", "dax", "tor", "ren", "mal", "skor", "vin", "drel", "lark",
    "thorn", "bric", "fen", "gar", "hol", "kest", "lun", "mor", "nash", "orv", "pell", "quin",
    "rask", "sel", "tav", "ul", "vark", "wex", "yor", "zell",
]
SURNAME_ENDINGS = ["brook", "by", "en", "ett", "ford", "ik", "ley", "mont", "ow", "ski", "ton", "well"]
ORG_SYLLABLES = [
    "tes", "sa", "line", "cor", "vane", "ael", "bri", "dor", "eth", "fal", "gry", "hel",
    "ith", "jun", "kal", "lum", "mer", "nov", "ost", "pra", "qua", "riv", "sol", "tyr",
]
ORG_SUFFIXES = [
    "Labs", "Institute", "Systems", "Group", "Works", "Analytics", "Foundry", "Collective",
    "Dynamics", "Partners", "Studios", "Robotics",
]

FACT1 = "{a}'s father is {b}."
FACT2 = "{b} works at {c}."
Q_HOP1 = "Who is {a}'s father?"
Q_HOP2 = "Where does {b} work?"
Q_TWO_HOP = "Where does {a}'s father work?"


def make_person_pools():
    """Return (all people, male people); fathers are sampled from the male pool."""
    surnames = [s.capitalize() + e for s, e in itertools.product(SURNAME_SYLLABLES, SURNAME_ENDINGS)]
    males = [f"{f} {s}" for f, s in itertools.product(MALE_NAMES, surnames)]
    females = [f"{f} {s}" for f, s in itertools.product(FEMALE_NAMES, surnames)]
    return males + females, males


def make_org_pool():
    prefixes = [(a + b).capitalize() for a, b in itertools.permutations(ORG_SYLLABLES, 2)]
    return [f"{p} {s}" for p, s in itertools.product(prefixes, ORG_SUFFIXES)]


def build_item(rng, persons, males, orgs, target_org, n_distractors, pos_pair, order):
    # Every person and org in a document is unique, so each question has exactly one answer.
    fathers = rng.sample(males, 1 + n_distractors)
    taken = set(fathers)
    children = []
    while len(children) < 1 + n_distractors:
        p = rng.choice(persons)
        if p not in taken:
            taken.add(p)
            children.append(p)
    distractor_orgs = rng.sample([o for o in orgs if o != target_org], n_distractors)
    a, b = children[0], fathers[0]
    fact1, fact2 = FACT1.format(a=a, b=b), FACT2.format(b=b, c=target_org)

    sentences = []
    for i in range(n_distractors):
        da, db = children[1 + i], fathers[1 + i]
        sentences += [FACT1.format(a=da, b=db), FACT2.format(b=db, c=distractor_orgs[i])]
    rng.shuffle(sentences)

    # Insert the later fact first so the earlier index stays valid.
    first, second = (fact1, fact2) if order == "forward" else (fact2, fact1)
    n = len(sentences)
    idx_first, idx_second = round(pos_pair[0] * n), round(pos_pair[1] * n)
    sentences.insert(idx_second, second)
    sentences.insert(idx_first, first)

    context, spans = "", {}
    for s in sentences:
        if context:
            context += " "
        if s in (fact1, fact2):
            spans["fact1_span" if s == fact1 else "fact2_span"] = [len(context), len(context) + len(s)]
        context += s

    return {
        "context": context,
        "hop1": {"q": Q_HOP1.format(a=a), "a": b},
        "hop2": {"q": Q_HOP2.format(b=b), "a": target_org},
        "two_hop": {"q": Q_TWO_HOP.format(a=a), "a": target_org},
        **spans,
        "entities": {"a": a, "b": b, "c": target_org},
        "config": {"n_distractors": n_distractors, "pos_pair": list(pos_pair), "order": order},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-items", type=int, default=100, help="items per configuration")
    parser.add_argument("--n-distractors", type=int, nargs="+", default=[25, 50, 100, 200],
                        help="number of distractor chains (controls length; ~20 tokens per chain)")
    parser.add_argument("--pos-pairs", nargs="+", default=["0.1,0.9", "0.1,0.5", "0.5,0.9", "0.45,0.55"],
                        help="relative positions of the earlier and later supporting fact")
    parser.add_argument("--orders", nargs="+", default=["forward", "reverse"], choices=["forward", "reverse"],
                        help="forward: fact1 (father) comes before fact2 (works at)")
    parser.add_argument("--tokenizer", default=None, help="optional HF tokenizer to record n_tokens per item")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="data/two_hop.jsonl")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    (persons, males), orgs = make_person_pools(), make_org_pool()
    pos_pairs = [tuple(sorted(float(x) for x in p.split(","))) for p in args.pos_pairs]
    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # Cycle through a shuffled org pool so every answer is used about equally often.
    answer_cycle = itertools.cycle(rng.sample(orgs, len(orgs)))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    count, token_counts = 0, {n_d: [] for n_d in args.n_distractors}
    with open(args.out, "w") as f:
        for n_d, pos_pair, order in itertools.product(args.n_distractors, pos_pairs, args.orders):
            for _ in range(args.n_items):
                item = build_item(rng, persons, males, orgs, next(answer_cycle), n_d, pos_pair, order)
                item["id"] = count
                if tokenizer is not None:
                    item["n_tokens"] = len(tokenizer(item["context"])["input_ids"])
                    token_counts[n_d].append(item["n_tokens"])
                f.write(json.dumps(item) + "\n")
                count += 1
    print(f"wrote {count} items to {args.out} ({len(persons)} person names, {len(orgs)} org names available)")
    if tokenizer is not None:
        for n_d in args.n_distractors:
            print(f"  {n_d} distractors -> {round(sum(token_counts[n_d]) / len(token_counts[n_d]))} tokens on average")


if __name__ == "__main__":
    main()
