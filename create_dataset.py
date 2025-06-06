import os
import json
import string
import random
import argparse

def normalize(text):
    return (
        text.lower()
        .strip()
        .strip(string.punctuation)
        .replace("’", "'")
        .replace("“", '"')
        .replace("”", '"')
    )

def parse_txt_file(input_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()

    title = ""
    paragraphs = []
    current_paragraph = ""
    current_qas = []
    mode = None

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("# Title"):
            mode = "title"
            continue
        elif line.startswith("# Paragraph"):
            if current_paragraph:
                paragraphs.append((current_paragraph.strip(), current_qas))
                current_paragraph = ""
                current_qas = []
            mode = "paragraph"
            continue
        elif line.startswith("# QA"):
            mode = "qa"
            continue

        if mode == "title":
            title = line
        elif mode == "paragraph":
            current_paragraph += " " + line
        elif mode == "qa" and "|||" in line:
            q, a = line.split("|||", 1)
            current_qas.append((q.strip(), a.strip()))

    if current_paragraph:
        paragraphs.append((current_paragraph.strip(), current_qas))

    return title, paragraphs

def create_sbert_jsonl_paragraph_level(input_file, output_file, add_negatives=True, negative_ratio=1):
    title, paragraphs = parse_txt_file(input_file)
    positives = []

    for paragraph, qa_pairs in paragraphs:
        for question, answer in qa_pairs:
            if normalize(answer) in normalize(paragraph):
                positives.append({
                    "sentence1": question,
                    "sentence2": paragraph.strip(),
                    "label": 1
                })

    negatives = []
    if add_negatives:
        questions = [p["sentence1"] for p in positives]
        paragraphs_only = [p["sentence2"] for p in positives]
        for _ in range(len(positives) * negative_ratio):
            q = random.choice(questions)
            p = random.choice(paragraphs_only)
            if not any(pair["sentence1"] == q and pair["sentence2"] == p for pair in positives):
                negatives.append({
                    "sentence1": q,
                    "sentence2": p,
                    "label": 0
                })

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in positives + negatives:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return len(positives), len(negatives)

def process_path(input_path, output_folder, add_negatives=True, negative_ratio=1):
    if os.path.isfile(input_path):
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_file = os.path.join(output_folder, base_name + ".jsonl")
        pos, neg = create_sbert_jsonl_paragraph_level(input_path, output_file, add_negatives, negative_ratio)
        print(f"✅ Processed '{input_path}': {pos} positives, {neg} negatives.")
    elif os.path.isdir(input_path):
        txt_files = [f for f in os.listdir(input_path) if f.endswith(".txt")]
        os.makedirs(output_folder, exist_ok=True)
        for txt_file in txt_files:
            input_file = os.path.join(input_path, txt_file)
            base_name = os.path.splitext(txt_file)[0]
            output_file = os.path.join(output_folder, base_name + ".jsonl")
            pos, neg = create_sbert_jsonl_paragraph_level(input_file, output_file, add_negatives, negative_ratio)
            print(f"✅ Processed '{txt_file}': {pos} positives, {neg} negatives.")
        print(f"🎉 Finished processing {len(txt_files)} files.")
    else:
        print(f"❌ Error: '{input_path}' is not a file or directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .txt QA modules to SBERT JSONL format using full paragraphs.")
    parser.add_argument("input", help="Path to .txt file or folder containing .txt files")
    parser.add_argument("--output", default="datasets/sbert_jsonl", help="Output folder for .jsonl files")
    parser.add_argument("--no-negatives", action="store_true", help="Disable adding random negative pairs")
    parser.add_argument("--neg-ratio", type=int, default=1, help="Ratio of negatives to positives")

    args = parser.parse_args()
    process_path(args.input, args.output, not args.no_negatives, args.neg_ratio)

