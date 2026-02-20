from typing import Dict
from tqdm import tqdm

from io_utils import read_jsonl, write_jsonl
from verifier import score_structural_violations, exact_match
from omnia_posthoc import run_omnia

def index_by_id(rows):
    return {r["id"]: r for r in rows}

def main():
    gsm = read_jsonl("data/gsm8k.jsonl")
    base = read_jsonl("runs/baseline.jsonl")

    gsm_i = index_by_id(gsm)
    base_i = index_by_id(base)

    ids = [r["id"] for r in gsm if r["id"] in base_i]

    acc_n = 0
    svr_n = 0
    total = 0

    for _id in ids:
        total += 1
        gold = gsm_i[_id]["answer"]
        comp = base_i[_id]["completion"]

        v = score_structural_violations(comp)

        if not v["passed"]:
            svr_n += 1

        if exact_match(gold, v["final"]):
            acc_n += 1

    acc_base = acc_n / total if total else 0.0
    svr_base = svr_n / total if total else 0.0

    filtered = []
    acc2_n = 0
    svr2_n = 0
    total2 = 0

    for _id in tqdm(ids, desc="OMNIA post-hoc"):
        gold = gsm_i[_id]["answer"]
        row = base_i[_id].copy()
        comp = row["completion"]

        v = score_structural_violations(comp)
        o = run_omnia(comp)

        row["omnia"] = o
        row["final"] = v["final"]
        filtered.append(row)

        total2 += 1

        if not o["pass"]:
            svr2_n += 1

        if exact_match(gold, v["final"]):
            acc2_n += 1

    acc_omnia = acc2_n / total2 if total2 else 0.0
    svr_omnia = svr2_n / total2 if total2 else 0.0

    write_jsonl("runs/omnia_filtered.jsonl", filtered)

    print("=== REPORT ===")
    print(f"Samples: {total}")
    print(f"ACC baseline: {acc_base:.4f}")
    print(f"SVR baseline: {svr_base:.4f}")
    print(f"ACC posthoc : {acc_omnia:.4f}")
    print(f"SVR posthoc : {svr_omnia:.4f}")
    print(f"Delta SVR   : {(svr_base - svr_omnia):.4f}")
    print("Saved: runs/omnia_filtered.jsonl")

if __name__ == "__main__":
    main()