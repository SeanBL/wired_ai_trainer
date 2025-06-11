import os
import json
import argparse

def merge_jsonl_files(input_dir, output_file):
    merged = []
    seen_pairs = set()

    for fname in os.listdir(input_dir):
        if not fname.endswith(".jsonl"):
            continue
        path = os.path.join(input_dir, fname)
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                key = (item["sentence1"].strip().lower(), item["sentence2"].strip().lower())
                if key not in seen_pairs:
                    merged.append(item)
                    seen_pairs.add(key)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in merged:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"✅ Merged {len(merged)} unique QA pairs into '{output_file}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge multiple .jsonl files into one SBERT training file.")
    parser.add_argument("--input", default="datasets/sbert_augmented", help="Directory with .jsonl files")
    parser.add_argument("--output", default="datasets/sbert_train/healthmap_combined.jsonl", help="Output path")

    args = parser.parse_args()
    merge_jsonl_files(args.input, args.output)
