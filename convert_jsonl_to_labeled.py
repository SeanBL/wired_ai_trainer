import os
import json
import argparse

def load_curated_jsonl(input_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = [json.loads(line.strip()) for line in f if line.strip()]
    return lines

def save_sbert_jsonl(data, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')

def process_file(input_file, output_file):
    data = load_curated_jsonl(input_file)

    positives = []
    for item in data:
        q = item["sentence1"].strip()
        a = item["sentence2"].strip()
        positives.append({
            "sentence1": q,
            "sentence2": a,
            "label": 1.0
        })

    save_sbert_jsonl(positives, output_file)
    print(f"✅ Processed {len(positives)} labeled pairs into '{output_file}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Claude-style QA JSONL into SBERT labeled format.")
    parser.add_argument("input", help="Path to input .jsonl file with sentence1/sentence2 pairs")
    parser.add_argument("--output", help="(Optional) Output file path. Default: datasets/sbert_labeled/<filename>_labeled.jsonl")

    args = parser.parse_args()

    if args.output:
        output_path = args.output
    else:
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        output_path = os.path.join("datasets/sbert_labeled", f"{base_name}_labeled.jsonl")

    process_file(args.input, output_path)
