from typing import Dict
from verifier import score_structural_violations

def run_omnia(completion: str) -> Dict:
    """
    Placeholder deterministico: oggi usa il verifier come gate strutturale.
    Sostituibile con OMNIA reale (Ω score + ICE envelope + flags).
    """
    v = score_structural_violations(completion)
    score = 1.0 if v["passed"] else 0.0
    return {
        "score": score,
        "pass": v["passed"],
        "flags": v["flags"],
    }