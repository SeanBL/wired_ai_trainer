import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from colorama import Fore, Style
from datetime import datetime

def list_available_models(models_dir="models"):
    models = [f for f in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, f))]
    return models

def load_paragraphs_from_squad(squad_path):
    with open(squad_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    paragraphs = []
    for topic in data["data"]:
        for para in topic["paragraphs"]:
            text = para["context"].strip()
            if text not in paragraphs:
                paragraphs.append(text)
    return paragraphs

def generate_embeddings(paragraphs, model_path):
    print(Fore.YELLOW + f"📥 Loading model from: {model_path}")
    model = SentenceTransformer(model_path)
    embeddings = model.encode(paragraphs, show_progress_bar=True, convert_to_numpy=True)
    return embeddings, paragraphs

def save_outputs(embeddings, paragraphs, out_dir="datasets/embeddings", version_tag=None):
    os.makedirs(out_dir, exist_ok=True)

    if not version_tag:
        version_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    np.save(f"{out_dir}/sbert_embeddings_{version_tag}.npy", embeddings)
    
    metadata = [{"id": i, "text": p} for i, p in enumerate(paragraphs)]
    with open(f"{out_dir}/sbert_metadata_{version_tag}.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(Style.BRIGHT + f"\n✅ Saved embeddings to: {out_dir}/sbert_embeddings_{version_tag}.npy")
    print(Fore.CYAN + f"✅ Saved metadata to: {out_dir}/sbert_metadata_{version_tag}.json")

def main():
    squad_path = "datasets/squad/first_aid_for_cuts_and_scrapes.json"
    print(Fore.YELLOW + f"🔍 Loading paragraphs from: {squad_path}")
    
    paragraphs = load_paragraphs_from_squad(squad_path)
    print(Fore.YELLOW + f"🔢 Generating embeddings for {len(paragraphs)} unique paragraphs...\n")

    # 📌 Let user choose the model version
    available_models = list_available_models()
    print(Fore.CYAN + "📦 Available Models:")
    for idx, model_name in enumerate(available_models):
        print(f"  [{idx}] {model_name}")

    model_choice = input("\n➡️  Enter the number corresponding to the model you want to use: ").strip()
    try:
        model_choice = int(model_choice)
        selected_model = os.path.join("models", available_models[model_choice])
    except (ValueError, IndexError):
        print(Fore.RED + "❌ Invalid selection. Exiting.")
        return

    version_tag = input("\n📌 Enter a version tag for the embeddings (or leave blank for timestamp): ").strip() or None

    embeddings, passages = generate_embeddings(paragraphs, selected_model)
    save_outputs(embeddings, passages, version_tag=version_tag)

if __name__ == "__main__":
    main()

