import os
import gradio as gr
from sentence_transformers import SentenceTransformer, util
import numpy as np
import json

# Load model and metadata
base_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_path, "models", "sbert_first_aid_finetuned_v1.0.0")
model = SentenceTransformer(model_path)
embedding_path = os.path.join(base_path, "datasets", "embeddings", "sbert_embeddings_1.0.0.npy")
metadata_path = os.path.join(base_path, "datasets", "embeddings", "sbert_metadata_1.0.0.json")
embeddings = np.load(embedding_path)
with open(metadata_path, "r", encoding="utf-8") as f:
    metadata = json.load(f)

#Format results for display
def format_results_for_display(results):
    formatted_blocks = []

    for item in results:
        if "(Score:" in item:
            content, score = item.rsplit("(Score:", 1)
            score = score.strip(" )")
        else:
            content = item
            score = "N/A"

        content = content.replace("\\n", "\n").strip()
        content = content.lstrip("🔹").strip()

        lines = content.splitlines()
        if lines:
            heading = lines[0]
            body = "<br>".join(lines[1:]).strip()
        else:
            heading = "Answer"
            body = content

        formatted = f"""
            <div style="margin-bottom: 1.5em;">
                <strong style="color: #0077cc;">{heading}</strong><br>
                {body}<br>
                <span style="color: green; font-weight: bold;">Score: {score}</span>
            </div>
        """
        formatted_blocks.append(formatted)

    return "<hr>".join(formatted_blocks)

# Extract texts
texts = [item["text"] for item in metadata]

def semantic_search(query, top_k=5):
    query_emb = model.encode(query, convert_to_tensor=True)
    scores = util.cos_sim(query_emb, embeddings)[0].cpu().numpy()
    top_idx = np.argsort(scores)[-top_k:][::-1]
    raw_results = [f"🔹 {texts[i]}\n(Score: {scores[i]:.3f})" for i in top_idx]
    return format_results_for_display(raw_results)

iface = gr.Interface(
    fn=semantic_search,
    inputs=gr.Textbox(placeholder="Ask a health question...", label="Your Question"),
    outputs=gr.HTML(label="Top Matching Answers"),
    title="SBERT Health Assistant",
    description="Ask any question related to first aid or health training modules."
)

if __name__ == "__main__":
    iface.launch(share=False)
