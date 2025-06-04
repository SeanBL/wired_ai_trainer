import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer, util
from colorama import Fore, Style
import torch

def list_available_embeddings(embeddings_dir="datasets/embeddings"):
    return [f for f in os.listdir(embeddings_dir) if f.startswith("sbert_embeddings") and f.endswith(".npy")]

def load_embeddings_and_metadata(embeddings_file, metadata_file):
    embeddings = np.load(embeddings_file)
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    return embeddings, metadata

def semantic_search(query, model_path, embeddings, metadata, top_k=5):
    model = SentenceTransformer(model_path)
    query_embedding = model.encode(query, convert_to_tensor=True)
    similarities = util.cos_sim(query_embedding, embeddings)[0]
    top_results = torch.topk(similarities, k=top_k)

    print(Fore.YELLOW + "\n📋 Top Matches:\n")
    for score, idx in zip(top_results.values, top_results.indices):
        paragraph = metadata[int(idx)]["text"]
        print(Fore.GREEN + f"Score: {score.item():.4f}")
        print(paragraph + "\n")

def main():
    embeddings_dir = "datasets/embeddings"
    models_dir = "models"

    # Select embeddings
    available_embeddings = list_available_embeddings(embeddings_dir)
    print(Fore.CYAN + "📦 Available Embedding Files:")
    for idx, emb_file in enumerate(available_embeddings):
        print(f"  [{idx}] {emb_file}")

    emb_choice = input("\n➡️  Enter the number corresponding to the embeddings to use: ").strip()
    try:
        emb_choice = int(emb_choice)
        emb_file = available_embeddings[emb_choice]
        version_tag = emb_file.replace("sbert_embeddings_", "").replace(".npy", "")
    except (ValueError, IndexError):
        print(Fore.RED + "❌ Invalid selection. Exiting.")
        return

    embeddings_path = os.path.join(embeddings_dir, emb_file)
    metadata_path = os.path.join(embeddings_dir, f"sbert_metadata_{version_tag}.json")
    embeddings, metadata = load_embeddings_and_metadata(embeddings_path, metadata_path)

    # Select model
    available_models = [f for f in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, f))]
    print(Fore.CYAN + "\n📦 Available Models:")
    for idx, model_name in enumerate(available_models):
        print(f"  [{idx}] {model_name}")

    model_choice = input("\n➡️  Enter the number corresponding to the model to use: ").strip()
    try:
        model_choice = int(model_choice)
        model_path = os.path.join(models_dir, available_models[model_choice])
    except (ValueError, IndexError):
        print(Fore.RED + "❌ Invalid selection. Exiting.")
        return

    top_k = input("\n📌 Enter how many top results to show (default 5): ").strip()
    try:
        top_k = int(top_k) if top_k else 5
    except ValueError:
        top_k = 5

    print(Fore.YELLOW + "\n💬 Enter your question below (type 'exit' to quit):")
    while True:
        query = input(Style.BRIGHT + "\n📝 Enter a question: ").strip()
        if query.lower() == "exit":
            print(Fore.GREEN + "\n👋 Exiting Semantic Search. Goodbye!")
            break

        semantic_search(query, model_path, embeddings, metadata, top_k=top_k)

if __name__ == "__main__":
    main()

