import os
import json
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

# Load SBERT model once
model = SentenceTransformer("all-MiniLM-L6-v2")

# Thresholds
SIMILARITY_THRESHOLD = 0.70
REDUNDANCY_THRESHOLD = 0.90

INPUT_DIR = "datasets/sbert_augmented"
REPORT_DIR = "validation_reports"
os.makedirs(REPORT_DIR, exist_ok=True)

def load_jsonl(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f if line.strip()]

def validate_file(file_path, file_name):
    data = load_jsonl(file_path)
    results = []
    redundant = []
    duplicates = []

    # Validate similarity
    for i, item in enumerate(tqdm(data, desc=f"🔍 {file_name}"), 1):
        q, a = item["sentence1"], item["sentence2"]
        q_emb = model.encode(q, convert_to_tensor=True)
        a_emb = model.encode(a, convert_to_tensor=True)
        score = util.cos_sim(q_emb, a_emb).item()

        results.append({
            "index": i,
            "question": q,
            "answer": a,
            "similarity": round(score, 3),
            "is_supported": score >= SIMILARITY_THRESHOLD
        })

    # Detect redundant questions
    questions = [item["sentence1"] for item in data]
    embeddings = model.encode(questions, convert_to_tensor=True)

    for i in range(len(questions)):
        for j in range(i + 1, len(questions)):
            sim = util.cos_sim(embeddings[i], embeddings[j]).item()
            if sim >= REDUNDANCY_THRESHOLD:
                redundant.append({
                    "q1_index": i + 1,
                    "q1": questions[i],
                    "q2_index": j + 1,
                    "q2": questions[j],
                    "similarity": round(sim, 3)
                })

    # Detect exact duplicates
    seen = set()
    for i, item in enumerate(data):
        key = (item["sentence1"].strip().lower(), item["sentence2"].strip().lower())
        if key in seen:
            duplicates.append((i + 1, item["sentence1"]))
        else:
            seen.add(key)

    return results, redundant, duplicates

def save_report(filename, results, redundant, duplicates):
    with open(os.path.join(REPORT_DIR, f"{filename}_validation.txt"), 'w', encoding='utf-8') as f:
        supported = [r for r in results if r["is_supported"]]
        skipped = [r for r in results if not r["is_supported"]]

        f.write(f"📊 Validation Report for {filename}\n")
        f.write(f"Total QA pairs: {len(results)}\n")
        f.write(f"✅ Supported (similarity ≥ {SIMILARITY_THRESHOLD}): {len(supported)}\n")
        f.write(f"❌ Skipped (below threshold): {len(skipped)}\n")
        f.write(f"⚠️ Redundant questions (≥ {REDUNDANCY_THRESHOLD}): {len(redundant)}\n")
        f.write(f"📎 Duplicates: {len(duplicates)}\n\n")

        if skipped:
            f.write("❌ Low-similarity QAs:\n")
            for r in skipped:
                f.write(f"\n  #{r['index']} (Sim: {r['similarity']})\n  Q: {r['question']}\n  A: {r['answer']}\n")

        if redundant:
            f.write("\n⚠️ Redundant Questions:\n")
            for r in redundant:
                f.write(f"\n  Q{r['q1_index']} ↔ Q{r['q2_index']} (Sim: {r['similarity']})\n")
                f.write(f"    Q1: {r['q1']}\n    Q2: {r['q2']}\n")

        if duplicates:
            f.write("\n📎 Duplicate QA Pairs:\n")
            for i, q in duplicates:
                f.write(f"  Duplicate at #{i}: {q}\n")

def run_batch_validation():
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".jsonl")]
    if not files:
        print("❌ No .jsonl files found in", INPUT_DIR)
        return

    print(f"🔎 Validating {len(files)} files from {INPUT_DIR}...\n")
    for file in files:
        file_path = os.path.join(INPUT_DIR, file)
        results, redundant, duplicates = validate_file(file_path, file)
        save_report(os.path.splitext(file)[0], results, redundant, duplicates)
        print(f"✅ Finished: {file} — {len(results)} QAs validated.\n")

    print(f"\n📁 All reports saved in: {REPORT_DIR}/")

if __name__ == "__main__":
    run_batch_validation()
