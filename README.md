# kv-compression-multihop

How does KV cache compression (eviction and quantization) affect a language model's ability to combine
facts across a long context? Each test item pairs two single-hop questions with the two-hop question
that composes them, all answered from the **same** compressed cache.

```
fact1:   "Liam Vosk's father is Oren Halbrook."
fact2:   "Oren Halbrook works at Tessaline Labs."
hop1:    "Who is Liam Vosk's father?"            -> Oren Halbrook
hop2:    "Where does Oren Halbrook work?"        -> Tessaline Labs
two_hop: "Where does Liam Vosk's father work?"   -> Tessaline Labs
```

All entities are fictional, so answers must come from the context, not the model's memory.

## Files

| File | Purpose |
|---|---|
| `generate_data.py` | Builds the synthetic dataset (fictional people/orgs, same-template distractors, controlled fact positions and lengths). |
| `data/two_hop.jsonl` | Generated dataset: 3,200 items (4 lengths x 4 position pairs x 2 orders x 100), ~0.5k-4.4k tokens. |
| `eval_kv.py` | Prefills the document, optionally compresses the KV cache, answers hop1/hop2/two_hop from that cache. |
| `run_eval.sbatch` | SLURM wrapper for `eval_kv.py` (UMIACS Nexus `class` partition). |
| `summarize.py` | Accuracy, independence baseline (acc_hop1 x acc_hop2), and composition-failure rate per condition. |
| `results/` | Pilot results (see below). |

Conditions in `eval_kv.py`:

- `full`: uncompressed cache
- `nocontext`: empty document, a sanity check that answers need the context
- `rand<r>`: kvpress `RandomPress`, evicting a fraction `r` of KV entries (e.g. `rand0.5`)
- `int<b>`: per-token round-to-nearest fake quantization of K and V to `b` bits (e.g. `int4`)

## Setup

Python 3.10-3.12 is recommended; 3.13 also works through the `standard-pipes` shim in `requirements.txt`.
A CUDA GPU is needed for the 7B model (~16 GB in bf16); a 24 GB card such as an RTX A5000 is enough.

```bash
git clone git@github.com:ZhichengZhou-1/kv-compression-multihop.git
cd kv-compression-multihop
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

On UMIACS Nexus, load Python first with `module load Python3/3.11.11` (in a login shell).

Models are downloaded from Hugging Face on first use, into `hf_cache/` in the repo when run through
`run_eval.sbatch`, or into `$HF_HOME` if you set it.

## Running

Regenerate the dataset (optional; `data/two_hop.jsonl` is already included and generation is seeded):

```bash
python generate_data.py --out data/two_hop.jsonl --tokenizer Qwen/Qwen2.5-1.5B-Instruct
```

Run directly on a GPU machine:

```bash
.venv/bin/python eval_kv.py --model Qwen/Qwen2.5-7B-Instruct --cot --max-new-tokens 160 \
    --n-per-config 2 --conditions full int8 --out results/test.jsonl
```

Or submit on Nexus from the repo root:

```bash
sbatch --gres=gpu:rtxa5000:1 \
    --export=ALL,EXTRA_ARGS="--model Qwen/Qwen2.5-7B-Instruct --cot --max-new-tokens 160" \
    run_eval.sbatch results/test.jsonl 2 full int8
```

Summarize:

```bash
python summarize.py results/pilot_7b_cot.jsonl
python summarize.py results/pilot_7b_cot.jsonl --by n_distractors
```

The output file is appended to and finished `(item, condition)` pairs are skipped, so a killed job can be
resubmitted with the same command.

## Pilot results (uncompressed cache, 64 items per run)

| Setup | hop1 | hop2 | two-hop | expected if independent |
|---|---|---|---|---|
| Qwen2.5-1.5B, direct answer | 0.52 | 0.63 | 0.05 | 0.32 |
| Qwen2.5-1.5B, CoT | 0.27 | 0.33 | 0.19 | 0.09 |
| Qwen2.5-7B, direct answer | 0.84 | 0.92 | 0.16 | 0.78 |
| **Qwen2.5-7B, CoT** | **0.92** | **0.88** | **0.83** | **0.81** |

- Only 7B with CoT solves two-hop. Without CoT, the model finds each fact but fails to combine them.
- int8 is essentially lossless. Per-token int4 collapses to 0% accuracy for every setup, likely because
  outlier key channels dominate each token's quantization range.

## Next steps

1. Main sweep on 7B + CoT: random eviction (10/25/50%) and quantization (int8/4/3/2).
2. Diagnose the int4 collapse: K-only vs V-only quantization, per-channel key quantization (KIVI-style).
3. Restoration: restore KV entries for fact 1, fact 2, both, and positions after the facts,
   against random restorations of the same size.
