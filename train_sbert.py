import json
import os
import datetime
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from checkpoint_manager import save_checkpoint

def load_training_data(jsonl_path):
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            data.append(InputExample(texts=[entry["sentence1"], entry["sentence2"]], label=float(entry["label"])))
    return data

def get_versioned_path(base_dir, base_name, version_tag=None):
    os.makedirs(base_dir, exist_ok=True)

    if version_tag:
        version_name = f"{base_name}_v{version_tag}"
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        version_name = f"{base_name}_{timestamp}"

    full_path = os.path.join(base_dir, version_name)
    os.makedirs(full_path, exist_ok=True)
    return full_path

def train_sbert(train_file, base_model="sentence-transformers/all-MiniLM-L6-v2", num_epochs=6, batch_size=32, version_tag=None):
    print(f"📂 Loading training data from: {train_file}")
    train_examples = load_training_data(train_file)
    print(f"✅ Loaded {len(train_examples)} training pairs")

    print(f"📥 Loading base model: {base_model}")
    model = SentenceTransformer(base_model)

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.CosineSimilarityLoss(model)

    # 📂 Get versioned output path
    output_path = get_versioned_path("models", "sbert_first_aid_finetuned", version_tag)

    print("🚀 Starting fine-tuning...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=num_epochs,
        warmup_steps=100,
        output_path=output_path
    )

    print(f"✅ Model saved to: {output_path}")

    # 📝 Save metadata + model using checkpoint_manager
    training_metadata = {
        "version": version_tag or "timestamped",
        "base_model": base_model,
        "dataset": os.path.basename(train_file),
        "epochs": num_epochs,
        "batch_size": batch_size,
        "notes": "SBERT fine-tuned on first aid Q&A dataset."
    }

    save_checkpoint(model, output_path, training_metadata)

if __name__ == "__main__":
    train_file = "datasets/sbert_train/first_aid_augmented.jsonl"

    BASE_MODELS = {
        "general_fast": "sentence-transformers/all-MiniLM-L6-v2",
        "paraphrase_optimized": "sentence-transformers/paraphrase-MiniLM-L6-v2",
        "bio_medical": "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
        "bio_medical_advanced": "pritamdeka/S-BioBERT-snli-mnli-scitail-mednli-stsb"
    }

    selected_model = BASE_MODELS["general_fast"]  # Default model

    # 👉 Optional: Add a manual version tag here if you want
    version_tag = input("📌 Enter a version tag (or leave blank for timestamp): ").strip() or None

    train_sbert(train_file, base_model=selected_model, version_tag=version_tag)


