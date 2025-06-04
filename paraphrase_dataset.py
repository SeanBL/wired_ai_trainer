from parrot import Parrot
import torch
import random
import json

# Load Parrot model
parrot = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5", use_gpu=torch.cuda.is_available())

# Load your training file (SBERT QA format)
with open("datasets/sbert_train/first_aid_for_cuts_and_scrapes.json", "r", encoding="utf-8") as f:
    original_data = json.load(f)

augmented_data = []

# Loop through each QA pair and generate paraphrases
for entry in original_data:
    question = entry["sentence1"]
    answer = entry["sentence2"]
    label = entry["label"]

    # Generate up to 2 paraphrases
    para_phrases = parrot.augment(input_phrase=question, use_gpu=torch.cuda.is_available(), max_return_phrases=2)

    # Add original
    augmented_data.append(entry)

    # Add paraphrased questions
    if para_phrases:
        for para, _ in para_phrases:
            augmented_data.append({
                "sentence1": para,
                "sentence2": answer,
                "label": label
            })

# Save augmented dataset
out_path = "datasets/sbert_train/first_aid_augmented.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(augmented_data, f, indent=2, ensure_ascii=False)

print(f"✅ Augmented dataset saved to: {out_path}")
