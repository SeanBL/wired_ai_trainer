import json
import sys
import os
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

# Constants
DEFAULT_DIR = "datasets/sbert_augmented"
SIMILARITY_THRESHOLD = 0.70
REDUNDANCY_THRESHOLD = 0.90

# Load SBERT model once
model = SentenceTransformer("all-MiniLM-L6-v2")

def list_jsonl_files(directory):
    return [f for f in os.listdir(directory) if f.endswith(".jsonl")]

def load_jsonl(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f if line.strip()]

def validate_pairs(data):
    print(f"\n🔍 Checking QA similarity (threshold = {SIMILARITY_THRESHOLD})...")
    flagged = []

    for i, item in enumerate(tqdm(data), 1):
        q, a = item["sentence1"], item["sentence2"]
        q_emb = model.encode(q, convert_to_tensor=True)
        a_emb = model.encode(a, convert_to_tensor=True)
        score = util.cos_sim(q_emb, a_emb).item()

        if score < SIMILARITY_THRESHOLD:
            flagged.append({
                "index": i,
                "question": q,
                "answer": a,
                "similarity": round(score, 3)
            })

    return flagged

def detect_redundant_questions(data):
    print(f"\n🔁 Checking for redundant questions (threshold = {REDUNDANCY_THRESHOLD})...")
    flagged = []
    questions = [item["sentence1"] for item in data]
    embeddings = model.encode(questions, convert_to_tensor=True)

    for i in range(len(questions)):
        for j in range(i + 1, len(questions)):
            sim = util.cos_sim(embeddings[i], embeddings[j]).item()
            if sim >= REDUNDANCY_THRESHOLD:
                flagged.append({
                    "q1_index": i + 1,
                    "q1": questions[i],
                    "q2_index": j + 1,
                    "q2": questions[j],
                    "similarity": round(sim, 3)
                })
    return flagged

def detect_duplicates(data):
    print(f"\n📑 Checking for exact duplicate QA pairs...")
    seen = set()
    duplicates = []
    for i, item in enumerate(data):
        pair_key = (item["sentence1"].strip().lower(), item["sentence2"].strip().lower())
        if pair_key in seen:
            duplicates.append((i + 1, item["sentence1"]))
        else:
            seen.add(pair_key)
    return duplicates

def pick_file(directory=DEFAULT_DIR):
    files = list_jsonl_files(directory)
    if not files:
        print(f"❌ No .jsonl files found in '{directory}'")
        return None

    print(f"\n📂 Available .jsonl files in '{directory}':")
    for idx, file in enumerate(files):
        print(f"  [{idx}] {file}")
    try:
        selection = int(input("\n➡️  Select a file by number: ").strip())
        return os.path.join(directory, files[selection])
    except (ValueError, IndexError):
        print("❌ Invalid selection.")
        return None

if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = pick_file()

    if not path or not os.path.isfile(path):
        print(f"❌ File not found or not selected: {path}")
        sys.exit(1)

    print(f"\n📂 Validating file: {path}")
    data = load_jsonl(path)

    # Run validators
    mismatched = validate_pairs(data)
    redundant = detect_redundant_questions(data)
    duplicates = detect_duplicates(data)

    # Report
    print(f"\n📊 Validation Summary for '{os.path.basename(path)}'")
    print(f"Total entries: {len(data)}")
    print(f"❗ Question–Answer mismatches (low similarity): {len(mismatched)}")
    print(f"⚠️ Redundant (similar) questions: {len(redundant)}")
    print(f"📎 Duplicate QA pairs: {len(duplicates)}")

    if mismatched:
        print("\n❗ Examples of Question–Answer mismatches:")
        for r in mismatched[:5]:
            print(f"\n  Entry #{r['index']} (Similarity: {r['similarity']})")
            print(f"  Q: {r['question']}")
            print(f"  A: {r['answer']}")

    if redundant:
        print("\n⚠️ Examples of redundant questions:")
        for r in redundant[:5]:
            print(f"\n  Entry #{r['q1_index']} and #{r['q2_index']} (Similarity: {r['similarity']})")
            print(f"  Q1: {r['q1']}")
            print(f"  Q2: {r['q2']}")

    if duplicates:
        print("\n📎 Examples of duplicate QA pairs:")
        for i, q in duplicates[:5]:
            print(f"  Duplicate at entry #{i}: {q}")
