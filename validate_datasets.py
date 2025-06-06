import os
import json

def validate_jsonl_file(filepath):
    valid_lines = 0
    invalid_lines = 0
    line_number = 0

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line_number += 1
            try:
                entry = json.loads(line)
                if not all(k in entry for k in ("sentence1", "sentence2", "label")):
                    raise ValueError("Missing required keys: sentence1, sentence2, or label.")
                if not isinstance(entry["label"], (int, float)) or entry["label"] not in [0, 1]:
                    raise ValueError("Label must be 0 or 1 (as int or float).")
                valid_lines += 1
            except Exception as e:
                print(f"❌ Line {line_number} in {filepath} — {str(e)}")
                invalid_lines += 1

    print(f"✅ {os.path.basename(filepath)}: {valid_lines} valid | {invalid_lines} invalid")

def main():
    folder = "datasets/sbert_jsonl"
    if not os.path.exists(folder):
        print(f"❌ Folder '{folder}' not found.")
        return

    jsonl_files = [f for f in os.listdir(folder) if f.endswith(".jsonl")]
    if not jsonl_files:
        print(f"❌ No JSONL files found in '{folder}'.")
        return

    print(f"🔍 Validating {len(jsonl_files)} JSONL dataset files...\n")
    for jsonl_file in jsonl_files:
        filepath = os.path.join(folder, jsonl_file)
        validate_jsonl_file(filepath)

    print("\n🎯 Validation complete.")

if __name__ == "__main__":
    main()