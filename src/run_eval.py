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

    # Baseline metrics (over all samples)
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

    # Post-hoc OMNIA (compute flags for all, then apply hard filter)
    all_rows = []
    kept_rows = []

    # Metrics over kept set only
    kept_total = 0
    kept_acc_n = 0
    kept_svr_n = 0  # should be 0 by construction if filter is "pass only"

    for _id in tqdm(ids, desc="OMNIA post-hoc"):
        gold = gsm_i[_id]["answer"]
        row = base_i[_id].copy()
        comp = row["completion"]

        v = score_structural_violations(comp)
        o = run_omnia(comp)

        row["omnia"] = o
        row["final"] = v["final"]
        all_rows.append(row)

        # Hard filter: keep only structurally passing samples
        if o["pass"]:
            kept_rows.append(row)
            kept_total += 1
            # SVR on kept should be zero if "pass" really means no violations
            if not v["passed"]:
                kept_svr_n += 1
            if exact_match(gold, v["final"]):
                kept_acc_n += 1

    # Save outputs
    write_jsonl("runs/omnia_filtered.jsonl", all_rows)
    write_jsonl("runs/omnia_kept.jsonl", kept_rows)

    # Metrics for kept set
    coverage = kept_total / total if total else 0.0
    acc_kept = kept_acc_n / kept_total if kept_total else 0.0
    svr_kept = kept_svr_n / kept_total if kept_total else 0.0

    print("=== REPORT ===")
    print(f"Samples: {total}")
    print(f"ACC baseline: {acc_base:.4f}")
    print(f"SVR baseline: {svr_base:.4f}")
    print("--- posthoc hard filter (kept set) ---")
    print(f"Coverage  : {coverage:.4f}")
    print(f"ACC kept  : {acc_kept:.4f}")
    print(f"SVR kept  : {svr_kept:.4f}")
    print("Saved: runs/omnia_filtered.jsonl")
    print("Saved: runs/omnia_kept.jsonl")

if __name__ == "__main__":
    main()