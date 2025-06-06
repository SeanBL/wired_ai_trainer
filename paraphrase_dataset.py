import os
import json
from parrot import Parrot
import torch

# Load Parrot model
parrot = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5", use_gpu=torch.cuda.is_available())

# Folder containing .jsonl files
input_dir = "datasets/sbert_jsonl"
output_dir = "datasets/sbert_train"
os.makedirs(output_dir, exist_ok=True)

# List available .jsonl files
jsonl_files = [f for f in os.listdir(input_dir) if f.endswith(".jsonl")]
if not jsonl_files:
    print("❌ No .jsonl files found in:", input_dir)
    exit()

print("📄 Available .jsonl files:")
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

# Generate paraphrases
for entry in original_data:
    question = entry["sentence1"]
    answer = entry["sentence2"]
    label = entry["label"]

    # Add original
    augmented_data.append(entry)

    # Try generating up to 2 paraphrases
    para_phrases = parrot.augment(input_phrase=question, use_gpu=torch.cuda.is_available(), max_return_phrases=2)

    if para_phrases:
        for para, _ in para_phrases:
            augmented_data.append({
                "sentence1": para,
                "sentence2": answer,
                "label": label
            })

# Save output as JSONL
with open(output_path, "w", encoding="utf-8") as f:
    for entry in augmented_data:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"\n✅ Augmented dataset saved to: {output_path}")
print(f"📊 Total pairs (original + paraphrased): {len(augmented_data)}")

