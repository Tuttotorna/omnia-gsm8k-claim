import re
from typing import Dict, List, Optional, Tuple

FINAL_PATTERNS = [
    re.compile(r"####\s*([-+]?\d+(\.\d+)?)"),
    re.compile(r"final\s*answer\s*[:\-]?\s*([-+]?\d+(\.\d+)?)", re.IGNORECASE),
    re.compile(r"answer\s*[:\-]?\s*([-+]?\d+(\.\d+)?)", re.IGNORECASE),
]

NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")

def extract_final_number(text: str) -> Optional[str]:
    for p in FINAL_PATTERNS:
        m = p.search(text)
        if m:
            return m.group(1)
    nums = NUM_RE.findall(text)
    if nums:
        last = nums[-1]
        if isinstance(last, tuple):
            return last[0]
        return last
    return None

def find_conflicting_finals(text: str) -> bool:
    finals = []
    for p in FINAL_PATTERNS:
        for m in p.finditer(text):
            finals.append(m.group(1))
    finals = list(dict.fromkeys(finals))
    return len(finals) >= 2

def basic_equation_checks(text: str) -> Tuple[int, int]:
    eq_re = re.compile(r"(\-?\d+(?:\.\d+)?)\s*([+\-*/])\s*(\-?\d+(?:\.\d+)?)\s*=\s*(\-?\d+(?:\.\d+)?)")
    checked = 0
    failed = 0
    for m in eq_re.finditer(text):
        a = float(m.group(1))
        op = m.group(2)
        b = float(m.group(3))
        c = float(m.group(4))

        if op == "+":
            v = a + b
        elif op == "-":
            v = a - b
        elif op == "*":
            v = a * b
        else:
            if b == 0:
                failed += 1
                checked += 1
                continue
            v = a / b

        checked += 1
        if abs(v - c) > 1e-6:
            failed += 1

    return checked, failed

def score_structural_violations(completion: str) -> Dict:
    flags: List[str] = []

    final = extract_final_number(completion)

    if final is None:
        flags.append("no_final_number")

    if find_conflicting_finals(completion):
        flags.append("conflicting_finals")

    checked, failed = basic_equation_checks(completion)
    if checked >= 1 and failed >= 1:
        flags.append("equation_mismatch")

    passed = len(flags) == 0

    return {
        "final": final,
        "flags": flags,
        "passed": passed,
        "eq_checked": checked,
        "eq_failed": failed,
    }

def exact_match(gold_answer: str, pred_final: Optional[str]) -> bool:
    if pred_final is None:
        return False

    m = re.search(r"####\s*([-+]?\d+(?:\.\d+)?)", gold_answer)
    if not m:
        return False

    return m.group(1) == pred_final