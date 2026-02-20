import json
import re
from pathlib import Path

RAW_PATH = Path("data/gsm8k_raw.jsonl")
OUT_PATH = Path("data/gsm8k.jsonl")

def extract_answer(text: str):
    # GSM8K answers usually end with #### 42
    match = re.search(r"####\s*([-+]?\d+)", text)
    if match:
        return match.group(1).strip()
    # fallback: last number in text
    nums = re.findall(r"[-+]?\d+", text)
    return nums[-1] if nums else None

def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError("Missing data/gsm8k_raw.jsonl")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    kept = 0

    with RAW_PATH.open("r", encoding="utf-8") as fin, \
         OUT_PATH.open("w", encoding="utf-8") as fout:

        for line in fin:
            total += 1
            obj = json.loads(line)

            question = obj.get("question", "").strip()
            answer_text = obj.get("answer", "")

            final_answer = extract_answer(answer_text)
            if final_answer is None:
                continue

            record = {
                "id": total,
                "question": question,
                "gold": final_answer,
            }

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1

    print("Conversion complete")
    print(f"Total raw: {total}")
    print(f"Kept: {kept}")
    print(f"Saved to: {OUT_PATH}")

if __name__ == "__main__":
    main()