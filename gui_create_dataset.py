import os
import json
import tkinter as tk
from tkinter import filedialog, messagebox

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
                    if qa_entry:
                        current_qas.append(qa_entry)

    if current_context and current_qas:
        paragraphs.append(create_paragraph(current_context, current_qas))

    return title, paragraphs

def create_dataset_from_file(file_path):
    title, paragraphs = parse_input_file(file_path)

    if not title or not paragraphs:
        messagebox.showerror("Error", "Missing title or paragraphs.")
        return

    dataset = create_dataset(title, paragraphs)

    folder = "datasets"
    os.makedirs(folder, exist_ok=True)

    filename = title.lower().replace(" ", "_") + ".json"
    filepath = os.path.join(folder, filename)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    messagebox.showinfo("Success", f"✅ Dataset created!\nSaved as '{filepath}'.")

def select_file():
    file_path = filedialog.askopenfilename(
        title="Select your input .txt file",
        filetypes=[("Text files", "*.txt")]
    )
    if file_path:
        create_dataset_from_file(file_path)

def main():
    root = tk.Tk()
    root.title("Dataset Creator")

    canvas = tk.Canvas(root, width=300, height=150)
    canvas.pack()

    button = tk.Button(root, text="Select .txt File to Create Dataset", command=select_file, height=2, width=30)
    canvas.create_window(150, 75, window=button)

    root.mainloop()

if __name__ == "__main__":
    main()
