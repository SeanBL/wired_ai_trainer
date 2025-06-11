import os
import json
from parrot import Parrot
import torch
from sentence_transformers import SentenceTransformer, util

# Load Parrot
parrot = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5", use_gpu=torch.cuda.is_available())

# Load SBERT for filtering
similarity_model = SentenceTransformer("all-MiniLM-L6-v2")
PARAPHRASE_SIMILARITY_THRESHOLD = 0.75

# Folder paths
input_dir = "datasets/sbert_labeled"
output_dir = "datasets/sbert_augmented"
os.makedirs(output_dir, exist_ok=True)

# List available .jsonl files
jsonl_files = [f for f in os.listdir(input_dir) if f.endswith(".jsonl")]
if not jsonl_files:
    print("❌ No .jsonl files found in:", input_dir)
    exit()

print("📄 Available labeled .jsonl files:")
for i, f in enumerate(jsonl_files):
    print(f"  [{i}] {f}")

# Ask user to pick a file
try:
    choice = int(input("\n➡️ Select a file to paraphrase by number: ").strip())
    input_file = jsonl_files[choice]
except (ValueError, IndexError):
    print("❌ Invalid choice.")
    exit()

input_path = os.path.join(input_dir, input_file)
base_name = os.path.splitext(input_file)[0]
output_path = os.path.join(output_dir, f"{base_name}_augmented.jsonl")

# Load dataset
with open(input_path, 'r', encoding='utf-8') as f:
    original_data = [json.loads(line.strip()) for line in f]

augmented_data = []

# Generate and filter paraphrases
for entry in original_data:
    question = entry["sentence1"]
    answer = entry["sentence2"]
    label = entry["label"]

    # Always include the original
    augmented_data.append(entry)

    para_phrases = parrot.augment(input_phrase=question, use_gpu=torch.cuda.is_available(), max_return_phrases=2)

    if para_phrases:
        original_emb = similarity_model.encode(question, convert_to_tensor=True)
        for para, _ in para_phrases:
            para_emb = similarity_model.encode(para, convert_to_tensor=True)
            sim_score = util.cos_sim(original_emb, para_emb).item()

            if sim_score >= PARAPHRASE_SIMILARITY_THRESHOLD:
                augmented_data.append({
                    "sentence1": para,
                    "sentence2": answer,
                    "label": label
                })
            else:
                print(f"\n❌ Skipped (similarity = {sim_score:.3f}):")
                print(f"  Original: {question}")
                print(f"  Paraphrased: {para}")

# Save result
with open(output_path, "w", encoding="utf-8") as f:
    for item in augmented_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"\n✅ Augmented dataset saved to: {output_path}")
print(f"📊 Total pairs (original + accepted paraphrases): {len(augmented_data)}")

