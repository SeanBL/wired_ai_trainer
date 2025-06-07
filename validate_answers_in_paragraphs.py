import nltk
nltk.download('punkt')
nltk.download('punkt_tab')  # <- this is what it's really asking for
from sentence_transformers import SentenceTransformer, util
from nltk.tokenize import sent_tokenize
import sys

# Load SBERT model once
model = SentenceTransformer("all-MiniLM-L6-v2")

# Thresholds
SIMILARITY_THRESHOLD = 0.7         # For QA validation
REDUNDANCY_THRESHOLD = 0.9         # For redundant question flagging

# Load QAs from structured file
def load_paragraphs_and_qas(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
    
    dataset = []
    paragraph = ""
    qas = []
    for line in lines:
        line = line.strip()
        if line.startswith("# Paragraph"):
            if paragraph and qas:
                dataset.append({"paragraph": paragraph.strip(), "qas": qas})
                qas = []
            paragraph = ""
        elif line.startswith("# QA"):
            continue
        elif "|||" in line:
            q, a = line.split("|||")
            qas.append((q.strip(), a.strip()))
        elif line:
            paragraph += " " + line.strip()
    
    if paragraph and qas:
        dataset.append({"paragraph": paragraph.strip(), "qas": qas})
    
    return dataset

# Validate whether each answer is supported by the paragraph
def validate(dataset):
    results = []
    for i, entry in enumerate(dataset, start=1):
        sentences = sent_tokenize(entry["paragraph"])
        for q, a in entry["qas"]:
            answer_emb = model.encode(a, convert_to_tensor=True)
            max_score = 0
            best_match = ""
            for sentence in sentences:
                sentence_emb = model.encode(sentence, convert_to_tensor=True)
                score = util.cos_sim(answer_emb, sentence_emb).item()
                if score > max_score:
                    max_score = score
                    best_match = sentence
            results.append({
                "Paragraph #": i,
                "Question": q,
                "Answer": a,
                "Best Matching Sentence": best_match,
                "Max Similarity": round(max_score, 3),
                "✅ Is Supported": max_score >= SIMILARITY_THRESHOLD
            })
    return results

# Flag redundant questions in each paragraph
def detect_redundant_questions(dataset):
    print(f"\n🔁 Checking for redundant questions (similarity ≥ {REDUNDANCY_THRESHOLD})...")
    for i, entry in enumerate(dataset, start=1):
        questions = [q for q, _ in entry["qas"]]
        if len(questions) < 2:
            continue
        embeddings = model.encode(questions, convert_to_tensor=True)
        for j in range(len(questions)):
            for k in range(j + 1, len(questions)):
                sim = util.cos_sim(embeddings[j], embeddings[k]).item()
                if sim >= REDUNDANCY_THRESHOLD:
                    print(f"\n⚠️ Redundant Qs in Paragraph {i} (Similarity: {round(sim, 3)})")
                    print(f"  Q1: {questions[j]}")
                    print(f"  Q2: {questions[k]}")

# Run validator
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Please provide a filename to validate.")
        sys.exit(1)

    filepath = sys.argv[1]
    print(f"\n🔍 Validating answers in: {filepath}")
    dataset = load_paragraphs_and_qas(filepath)
    result = validate(dataset)

    supported = [r for r in result if r["✅ Is Supported"]]
    skipped = [r for r in result if not r["✅ Is Supported"]]

    for r in skipped:
        print(f"\n⚠️ Skipped (not matched):\n  Q: {r['Question']}\n  A: {r['Answer']}\n  🔎 In Paragraph {r['Paragraph #']}")

    print(f"\n📊 Validation Complete:")
    print(f"✅ Answers passed threshold: {len(supported)}")
    print(f"❌ Answers skipped: {len(skipped)}")
    print(f"📄 Total QAs checked: {len(result)}")

    detect_redundant_questions(dataset)



