from parrot import Parrot
import torch
import json

# Load Parrot paraphraser
parrot = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5", use_gpu=torch.cuda.is_available())

# Load SBERT-style training data (jsonl)
input_path = "datasets/sbert_jsonl/first_aid_for_cuts_and_scrapes.jsonl"
output_path = "datasets/sbert_train/first_aid_augmented.jsonl"

augmented_data = []

with open(input_path, 'r', encoding='utf-8') as f:
    for line in f:
        item = json.loads(line)
        question = item["sentence1"]
        answer = item["sentence2"]
        label = item["label"]

        # Add original pair
        augmented_data.append(item)

        # Generate paraphrases of the question
        paraphrases = parrot.augment(
            input_phrase=question,
            use_gpu=torch.cuda.is_available(),
            max_return_phrases=2
        )

        if paraphrases:
            for para, _ in paraphrases:
                augmented_data.append({
                    "sentence1": para,
                    "sentence2": answer,
                    "label": label
                })

# Save to JSONL
with open(output_path, 'w', encoding='utf-8') as f:
    for item in augmented_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"✅ Augmented dataset saved to: {output_path}")
