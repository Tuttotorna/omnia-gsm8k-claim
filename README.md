# omnia-gsm8k-claim

Deterministic post-hoc evaluation harness for OMNIA on GSM8K.

## Claim
OMNIA as a post-hoc structural layer reduces Structural Violation Rate (SVR) on LLM outputs while preserving accuracy.

## What this repo does
- Loads GSM8K-style data (jsonl)
- Evaluates baseline completions
- Applies a deterministic post-hoc structural gate (OMNIA placeholder)
- Computes:
  - Accuracy (ACC)
  - Structural Violation Rate (SVR)
  - Delta SVR (baseline - posthoc)
- Saves a reproducible report file

## Structure
- data/gsm8k.jsonl → ground truth (questions + #### answer)
- runs/baseline.jsonl → model completions
- src/ → evaluation and structural verifier
- runs/omnia_filtered.jsonl → output with OMNIA flags

## How to run (local)
```bash
pip install -r requirements.txt
python src/run_eval.py