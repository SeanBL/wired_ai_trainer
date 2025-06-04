import os
import json
import sys
import random
from tqdm import tqdm  # NEW: for progress bar!

try:
    from colorama import init, Fore, Style
    init(autoreset=True)
except ImportError:
    print("Installing colorama package for colored outputs...")
    os.system('pip install colorama')
    from colorama import init, Fore, Style
    init(autoreset=True)

# (rest of your functions like create_qa_entry, create_paragraph, etc.)

def convert_squad_to_sbert(squad_path, sbert_output_path):
    """Convert SQuAD-format JSON to SBERT sentence-pair JSON format."""
    with open(squad_path, 'r', encoding='utf-8') as f:
        squad_data = json.load(f)

    sbert_data = []

    for topic in squad_data['data']:
        for paragraph in topic['paragraphs']:
            context = paragraph['context'].strip()
            for qa in paragraph['qas']:
                question = qa['question'].strip()
                sbert_data.append({
                    "sentence1": question,
                    "sentence2": context
                })

    with open(sbert_output_path, 'w', encoding='utf-8') as f:
        json.dump(sbert_data, f, indent=2, ensure_ascii=False)

    print(Fore.MAGENTA + f"🧠 SBERT-format dataset created: '{sbert_output_path}'")

def create_sbert_training_data(sbert_input_path, sbert_train_output_path, add_negatives=True, negative_ratio=1):
    with open(sbert_input_path, 'r', encoding='utf-8') as f:
        pairs = json.load(f)

    training_data = []

    # Add positive examples (label = 1)
    for pair in pairs:
        training_data.append({
            "sentence1": pair["sentence1"],
            "sentence2": pair["sentence2"],
            "label": 1
        })

    # Add random negative examples (label = 0)
    if add_negatives:
        questions = [p["sentence1"] for p in pairs]
        passages = [p["sentence2"] for p in pairs]

        for _ in range(len(pairs) * negative_ratio):
            q = random.choice(questions)
            p = random.choice(passages)
            if not any(e["sentence1"] == q and e["sentence2"] == p for e in training_data):
                training_data.append({
                    "sentence1": q,
                    "sentence2": p,
                    "label": 0
                })

    os.makedirs(os.path.dirname(sbert_train_output_path), exist_ok=True)

    with open(sbert_train_output_path, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)

    print(Fore.BLUE + f"🧪 SBERT training data saved: '{sbert_train_output_path}' (Total: {len(training_data)} pairs)")

def parse_input_file(input_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()

    title = ""
    paragraphs = []
    current_paragraph = ""
    current_qas = []

    total_qas = 0
    skipped_qas = 0

    mode = None
    for line in lines:
        line = line.strip()
        if line == "":
            continue

        if line.startswith("# Title"):
            mode = "title"
            continue
        elif line.startswith("# Paragraph"):
            if current_paragraph:
                paragraphs.append((current_paragraph.strip(), current_qas))
                current_paragraph = ""
                current_qas = []
            mode = "paragraph"
            continue
        elif line.startswith("# QA"):
            mode = "qa"
            continue

        if mode == "title":
            title = line.strip()
        elif mode == "paragraph":
            current_paragraph += " " + line.strip()
        elif mode == "qa":
            if "|||" in line:
                question, answer = line.split("|||", 1)
                current_qas.append((question.strip(), answer.strip()))
                total_qas += 1
            else:
                skipped_qas += 1

    # Append the last paragraph if needed
    if current_paragraph:
        paragraphs.append((current_paragraph.strip(), current_qas))

    return title, paragraphs, total_qas, skipped_qas

def create_dataset(title, paragraphs):
    data = {
        "data": [
            {
                "title": title,
                "paragraphs": []
            }
        ]
    }

    for para_text, qa_pairs in paragraphs:
        qa_entries = []
        for i, (question, answer) in enumerate(qa_pairs):
            normalized_paragraph = para_text.lower().strip()
            normalized_answer = answer.lower().strip()

            answer_start = normalized_paragraph.find(normalized_answer)
            if answer_start == -1:
                continue  # skip if answer not found in paragraph

            qa_entry = {
                "id": f"{title.lower().replace(' ', '_')}_{i}",
                "question": question,
                "answers": [
                    {
                        "text": answer,
                        "answer_start": answer_start
                    }
                ]
            }
            qa_entries.append(qa_entry)

        paragraph_entry = {
            "context": para_text,
            "qas": qa_entries
        }

        data["data"][0]["paragraphs"].append(paragraph_entry)

    return data


def process_single_file(input_file):
    title, paragraphs, total_qas, skipped_qas = parse_input_file(input_file)

    if not title or not paragraphs:
        print(Fore.RED + f"❌ Error: Missing title or paragraphs in '{input_file}'")
        return

    dataset = create_dataset(title, paragraphs)

    # Save SQuAD-format dataset
    squad_folder = os.path.join("datasets", "squad")
    os.makedirs(squad_folder, exist_ok=True)
    squad_filename = title.lower().replace(" ", "_") + ".json"
    squad_filepath = os.path.join(squad_folder, squad_filename)

    with open(squad_filepath, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(Fore.GREEN + f"✅ SQuAD dataset created: '{squad_filepath}'")
    print(Style.BRIGHT + f"📊 Report for '{title}': {total_qas - skipped_qas}/{total_qas} QA pairs successfully saved.")

    if skipped_qas > 0:
        print(Fore.YELLOW + f"⚠️  {skipped_qas} QA pairs skipped.")

     # Also generate SBERT-format dataset
    sbert_folder = os.path.join("datasets", "sbert")
    os.makedirs(sbert_folder, exist_ok=True)
    sbert_filename = title.lower().replace(" ", "_") + ".json"
    sbert_filepath = os.path.join(sbert_folder, sbert_filename)

    convert_squad_to_sbert(squad_filepath, sbert_filepath)

    # Generate SBERT training dataset with labels
    sbert_train_folder = os.path.join("datasets", "sbert_train")
    sbert_train_filepath = os.path.join(sbert_train_folder, sbert_filename)
    create_sbert_training_data(sbert_filepath, sbert_train_filepath)


def main():
    if len(sys.argv) != 2:
        print(Fore.RED + "❗ Usage: python create_dataset.py <input_file.txt> or <input_folder>")
        return

    input_path = sys.argv[1]

    if os.path.isfile(input_path):
        # Single file mode
        process_single_file(input_path)
    elif os.path.isdir(input_path):
        # Batch mode: process all .txt files in folder
        txt_files = [f for f in os.listdir(input_path) if f.endswith(".txt")]
        if not txt_files:
            print(Fore.RED + "❌ No .txt files found in the specified folder.")
            return

        print(Fore.CYAN + f"\n📁 Found {len(txt_files)} files. Starting batch processing...\n")

        # NEW: Wrap files with tqdm for progress bar
        for txt_file in tqdm(txt_files, desc="Building datasets", ncols=80):
            full_path = os.path.join(input_path, txt_file)
            process_single_file(full_path)

        print(Fore.CYAN + "\n🎉 Batch processing complete!")
    else:
        print(Fore.RED + f"❌ Error: '{input_path}' is neither a file nor a folder.")

if __name__ == "__main__":
    main()

