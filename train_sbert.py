import json
import os
import datetime
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from checkpoint_manager import save_checkpoint

def load_training_data(json_path):
    data = []
    with open(json_path, 'r', encoding='utf-8') as f:
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

def select_training_file(directory="datasets/sbert_train"):
    jsonl_files = [f for f in os.listdir(directory) if f.endswith(".jsonl")]
    if not jsonl_files:
        print("❌ No .jsonl training files found.")
        exit(1)

    print("\n📄 Available training files:")
    for idx, fname in enumerate(jsonl_files):
        print(f"  [{idx}] {fname}")

    choice = input("\n➡️  Select a file by number: ").strip()
    if not choice.isdigit() or int(choice) < 0 or int(choice) >= len(jsonl_files):
        print("❌ Invalid selection.")
        exit(1)

    return os.path.join(directory, jsonl_files[int(choice)])

def train_sbert(train_file, base_model="sentence-transformers/all-MiniLM-L6-v2", num_epochs=6, batch_size=32, version_tag=None):
    print(f"\n📂 Loading training data from: {train_file}")
    train_examples = load_training_data(train_file)
    print(f"✅ Loaded {len(train_examples)} training pairs")

    print(f"📥 Loading base model: {base_model}")
    model = SentenceTransformer(base_model)

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.CosineSimilarityLoss(model)

    output_path = get_versioned_path("models", "sbert_first_aid_finetuned", version_tag)

    print("🚀 Starting fine-tuning...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=num_epochs,
        warmup_steps=100,
        output_path=output_path
    )

    print(f"✅ Model saved to: {output_path}")

    training_metadata = {
        "version": version_tag or "timestamped",
        "base_model": base_model,
        "dataset": os.path.basename(train_file),
        "epochs": num_epochs,
        "batch_size": batch_size,
        "notes": "SBERT fine-tuned on healthcare Q&A dataset."
    }

    save_checkpoint(model, output_path, training_metadata)

if __name__ == "__main__":
    train_file = select_training_file("datasets/sbert_train")

    BASE_MODELS = {
        "general_fast": "sentence-transformers/all-MiniLM-L6-v2",
        "paraphrase_optimized": "sentence-transformers/paraphrase-MiniLM-L6-v2",
        "bio_medical": "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
        "bio_medical_advanced": "pritamdeka/S-BioBERT-snli-mnli-scitail-mednli-stsb"
    }

    print("\n💡 Choose model to fine-tune:")
    print("  [0] Base models")
    print("  [1] Previously fine-tuned models")

    mode_choice = input("\n➡️  Enter 0 or 1: ").strip()
    if mode_choice == "0":
        print("\n📆 Available base models:")
        for idx, key in enumerate(BASE_MODELS.keys()):
            print(f"  [{idx}] {key} -> {BASE_MODELS[key]}")

        base_choice = input("\n➡️  Select a base model by number: ").strip()
        base_keys = list(BASE_MODELS.keys())
        if not base_choice.isdigit() or int(base_choice) not in range(len(base_keys)):
            print("❌ Invalid selection.")
            exit(1)
        selected_model = BASE_MODELS[base_keys[int(base_choice)]]

    elif mode_choice == "1":
        model_dirs = [d for d in os.listdir("models") if os.path.isdir(os.path.join("models", d))]
        if not model_dirs:
            print("❌ No fine-tuned models found in 'models/' directory.")
            exit(1)
        print("\n📦 Available fine-tuned models:")
        for idx, model in enumerate(model_dirs):
            print(f"  [{idx}] {model}")

        model_choice = input("\n➡️  Select a fine-tuned model by number: ").strip()
        if not model_choice.isdigit() or int(model_choice) not in range(len(model_dirs)):
            print("❌ Invalid selection.")
            exit(1)
        selected_model = os.path.join("models", model_dirs[int(model_choice)])

    else:
        print("❌ Invalid mode selection.")
        exit(1)

    version_tag = input("📌 Enter a version tag (or leave blank for timestamp): ").strip() or None

    train_sbert(train_file, base_model=selected_model, version_tag=version_tag)




