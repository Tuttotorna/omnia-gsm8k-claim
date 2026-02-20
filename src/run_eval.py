from tqdm import tqdm

from io_utils import read_jsonl, write_jsonl
from verifier import score_structural_violations, exact_match
from omnia_posthoc import run_omnia


def index_by_id(rows):
    return {str(r["id"]): r for r in rows}


def safe_get_answer(row):
    # Support both "answer" and legacy "gold"
    if "answer" in row:
        return str(row["answer"]).strip()
    if "gold" in row:
        return str(row["gold"]).strip()
    return ""


def main():
    # Load data
    gsm = read_jsonl("data/gsm8k.jsonl")
    base = read_jsonl("runs/baseline.jsonl")

    if not gsm:
        raise ValueError("gsm8k.jsonl is empty or not loaded")
    if not base:
        raise ValueError("baseline.jsonl is missing or empty")

    gsm_i = index_by_id(gsm)
    base_i = index_by_id(base)

    # Only evaluate ids present in both sets
    ids = [i for i in gsm_i.keys() if i in base_i]
    total = len(ids)

    if total == 0:
        raise ValueError("No overlapping IDs between gsm8k and baseline")

    acc_base_n = 0
    svr_base_n = 0

    all_rows = []
    kept_rows = []
    kept_acc_n = 0
    kept_svr_n = 0

    for _id in tqdm(ids, desc="OMNIA post-hoc"):
        gold = safe_get_answer(gsm_i[_id])
        comp = base_i[_id].get("completion", "")

        # Structural verification (baseline)
        v_base = score_structural_violations(comp)
        if not v_base["passed"]:
            svr_base_n += 1
        if exact_match(gold, v_base["final"]):
            acc_base_n += 1

        # OMNIA post-hoc filtering
        o = run_omnia(comp)
        final_answer = o["final"]

        row = {
            "id": _id,
            "gold": gold,
            "baseline_completion": comp,
            "final": final_answer,
            "passed": o["pass"],
        }
        all_rows.append(row)

        # Kept set (structurally valid)
        if o["pass"]:
            kept_rows.append(row)

            v_kept = score_structural_violations(final_answer)
            if not v_kept["passed"]:
                kept_svr_n += 1
            if exact_match(gold, v_kept["final"]):
                kept_acc_n += 1

    # Metrics
    acc_base = acc_base_n / total if total else 0.0
    svr_base = svr_base_n / total if total else 0.0

    kept_total = len(kept_rows)
    coverage = kept_total / total if total else 0.0
    acc_kept = kept_acc_n / kept_total if kept_total else 0.0
    svr_kept = kept_svr_n / kept_total if kept_total else 0.0

    # Save outputs
    write_jsonl("runs/omnia_filtered.jsonl", all_rows)
    write_jsonl("runs/omnia_kept.jsonl", kept_rows)

    # Report (deterministic, structural)
    print("=== REPORT ===")
    print(f"Samples: {total}")
    print(f"ACC baseline: {acc_base:.4f}")
    print(f"SVR baseline: {svr_base:.4f}")
    print("--- posthoc hard filter (kept set) ---")
    print(f"Coverage : {coverage:.4f}")
    print(f"ACC kept : {acc_kept:.4f}")
    print(f"SVR kept : {svr_kept:.4f}")
    print("Saved: runs/omnia_filtered.jsonl")
    print("Saved: runs/omnia_kept.jsonl")

    # Structural invariants (objective validation, no narrative)
    assert total > 0, "Invariant failed: no samples evaluated"
    assert svr_kept <= svr_base + 1e-8, \
        "Invariant failed: posthoc increased structural violations"
    assert acc_kept >= acc_base - 1e-8, \
        "Invariant failed: posthoc decreased conditional accuracy"
    assert coverage < 1.0, \
        "Invariant failed: filter kept everything (no discrimination)"


if __name__ == "__main__":
    main()