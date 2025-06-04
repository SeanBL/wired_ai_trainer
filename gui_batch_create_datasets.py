import os
import json
import tkinter as tk
from tkinter import filedialog, messagebox
from tqdm import tqdm

def create_qa_entry(context, question, answer_text):
    answer_start = context.find(answer_text)
    if answer_start == -1:
        return None
    return {
        "question": question,
        "answers": [
            {
                "text": answer_text,
                "answer_start": answer_start
            }
        ]
    }

def create_paragraph(context, qa_list):
    return {
        "context": context,
        "qas": qa_list
    }

def create_dataset(title, paragraphs):
    return {
        "data": [
            {
                "title": title,
                "paragraphs": paragraphs
            }
        ]
    }

def parse_input_file(file_path):
    title = None
    paragraphs = []
    current_context = None
    current_qas = []
    skipped_qas = []
    total_qas = 0

    mode = None

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith("# Title"):
                mode = "title"
                continue
            if line.startswith("# Paragraph"):
                if current_context and current_qas:
                    paragraphs.append(create_paragraph(current_context, current_qas))
                current_context = ""
                current_qas = []
                mode = "paragraph"
                continue
            if line.startswith("# QA"):
                mode = "qa"
                continue

            if mode == "title":
                title = line
            elif mode == "paragraph":
                if current_context:
                    current_context += " " + line
                else:
                    current_context = line
            elif mode == "qa":
                if "|||" in line:
                    question, answer = line.split("|||", 1)
                    qa_entry = create_qa_entry(current_context, question.strip(), answer.strip())
                    total_qas += 1
                    if qa_entry:
                        current_qas.append(qa_entry)
                    else:
                        skipped_qas.append((question.strip(), answer.strip()))

    if current_context and current_qas:
        paragraphs.append(create_paragraph(current_context, current_qas))

    return title, paragraphs, total_qas, skipped_qas

def batch_create_datasets(folder_path):
    txt_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
    if not txt_files:
        messagebox.showerror("Error", "No .txt files found in the folder.")
        return

    folder = "datasets"
    os.makedirs(folder, exist_ok=True)

    skipped_report = []

    for txt_file in tqdm(txt_files, desc="Building datasets", ncols=80):
        input_path = os.path.join(folder_path, txt_file)
        title, paragraphs, total_qas, skipped_qas = parse_input_file(input_path)

        if not title or not paragraphs:
            skipped_report.append(f"❌ {txt_file} — Missing title or paragraphs.\n")
            continue

        dataset = create_dataset(title, paragraphs)

        filename = title.lower().replace(" ", "_") + ".json"
        filepath = os.path.join(folder, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)

        skipped_report.append(f"✅ {txt_file} — {total_qas - len(skipped_qas)}/{total_qas} QAs saved.")

        if skipped_qas:
            skipped_report.append(f"⚠️ Skipped QAs in {txt_file}:")
            for q, a in skipped_qas:
                skipped_report.append(f"    Q: {q}")

    # Save the skipped report
    report_path = os.path.join(folder, "build_report.txt")
    with open(report_path, 'w', encoding='utf-8') as report_file:
        report_file.write("\n".join(skipped_report))

    messagebox.showinfo("Done", f"✅ Batch completed!\nReport saved to 'datasets/build_report.txt'.")

def select_folder():
    folder_path = filedialog.askdirectory(
        title="Select Folder Containing .txt Files"
    )
    if folder_path:
        batch_create_datasets(folder_path)

def main():
    root = tk.Tk()
    root.title("Batch Dataset Creator")

    canvas = tk.Canvas(root, width=300, height=150)
    canvas.pack()

    button = tk.Button(root, text="Select Folder to Create Datasets", command=select_folder, height=2, width=30)
    canvas.create_window(150, 75, window=button)

    root.mainloop()

if __name__ == "__main__":
    main()
