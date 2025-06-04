import os
import json
import zipfile
import tkinter as tk
from tkinter import filedialog, messagebox
from tqdm import tqdm

# ───────────────────────────────
# 🔧 Reusable Functions
# ───────────────────────────────

def create_qa_entry(context, question, answer_text):
    answer_start = context.find(answer_text)
    if answer_start == -1:
        return None
    return {
        "question": question,
        "answers": [{"text": answer_text, "answer_start": answer_start}]
    }

def create_paragraph(context, qa_list):
    return {"context": context, "qas": qa_list}

def create_dataset(title, paragraphs):
    return {"data": [{"title": title, "paragraphs": paragraphs}]}

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
                current_context = (current_context + " " + line).strip() if current_context else line
            elif mode == "qa" and "|||" in line:
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

# ───────────────────────────────
# 🎯 Core Actions
# ───────────────────────────────

def create_single_dataset():
    file_path = filedialog.askopenfilename(title="Select a .txt module file", filetypes=[("Text files", "*.txt")])
    if not file_path:
        return

    title, paragraphs, total_qas, skipped_qas = parse_input_file(file_path)
    if not title or not paragraphs:
        messagebox.showerror("Error", "Missing title or paragraphs.")
        return

    os.makedirs("datasets", exist_ok=True)
    filename = title.lower().replace(" ", "_") + ".json"
    filepath = os.path.join("datasets", filename)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(create_dataset(title, paragraphs), f, indent=2, ensure_ascii=False)

    msg = f"✅ Created '{filename}'\n{total_qas - len(skipped_qas)}/{total_qas} QA saved"
    if skipped_qas:
        msg += f"\n⚠️ {len(skipped_qas)} QA(s) skipped"
    messagebox.showinfo("Done", msg)

def batch_create_datasets():
    folder_path = filedialog.askdirectory(title="Select folder with .txt files")
    if not folder_path:
        return

    os.makedirs("datasets", exist_ok=True)
    txt_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
    if not txt_files:
        messagebox.showerror("Error", "No .txt files found in the folder.")
        return

    skipped_report = []
    for txt_file in tqdm(txt_files, desc="Creating datasets", ncols=70):
        full_path = os.path.join(folder_path, txt_file)
        title, paragraphs, total_qas, skipped_qas = parse_input_file(full_path)

        if not title or not paragraphs:
            skipped_report.append(f"❌ {txt_file} — Missing title or paragraphs.")
            continue

        filename = title.lower().replace(" ", "_") + ".json"
        filepath = os.path.join("datasets", filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(create_dataset(title, paragraphs), f, indent=2, ensure_ascii=False)

        skipped_report.append(f"✅ {txt_file} — {total_qas - len(skipped_qas)}/{total_qas} QA saved.")
        if skipped_qas:
            skipped_report.append(f"⚠️ Skipped QAs:")
            for q, _ in skipped_qas:
                skipped_report.append(f"    Q: {q}")

    with open("datasets/build_report.txt", 'w', encoding='utf-8') as report:
        report.write("\n".join(skipped_report))

    messagebox.showinfo("Done", "✅ Batch complete!\nReport saved to datasets/build_report.txt.")

def validate_all_datasets():
    folder = "datasets"
    if not os.path.exists(folder):
        messagebox.showerror("Error", "No 'datasets' folder found.")
        return

    files = [f for f in os.listdir(folder) if f.endswith(".json")]
    if not files:
        messagebox.showerror("Error", "No .json files found in 'datasets'.")
        return

    report = []
    for file in files:
        try:
            with open(os.path.join(folder, file), 'r', encoding='utf-8') as f:
                data = json.load(f)
            if "data" not in data:
                raise Exception("Missing 'data' key")
            report.append(f"✅ {file} is valid")
        except Exception as e:
            report.append(f"❌ {file} is invalid: {str(e)}")

    with open(os.path.join(folder, "validation_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    messagebox.showinfo("Validation Done", "✅ Report saved to datasets/validation_report.txt")

def zip_all_datasets():
    folder = "datasets"
    if not os.path.exists(folder):
        messagebox.showerror("Error", "No 'datasets' folder found.")
        return

    files = [f for f in os.listdir(folder) if f.endswith(".json")]
    if not files:
        messagebox.showerror("Error", "No JSON files to zip.")
        return

    with zipfile.ZipFile("datasets.zip", 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in files:
            zipf.write(os.path.join(folder, file), arcname=file)

    messagebox.showinfo("Zipped", "✅ Datasets zipped into 'datasets.zip'.")

# ───────────────────────────────
# 🖼️ GUI Setup
# ───────────────────────────────

def main():
    root = tk.Tk()
    root.title("HealthMAP Dataset Studio")
    root.geometry("350x330")
    root.resizable(False, False)

    tk.Label(root, text="HealthMAP Dataset Studio", font=("Helvetica", 16, "bold")).pack(pady=10)

    tk.Button(root, text="📄 Create One Dataset", command=create_single_dataset, width=30, height=2).pack(pady=5)
    tk.Button(root, text="📂 Batch Create Datasets", command=batch_create_datasets, width=30, height=2).pack(pady=5)
    tk.Button(root, text="🧪 Validate Datasets", command=validate_all_datasets, width=30, height=2).pack(pady=5)
    tk.Button(root, text="📦 Zip All Datasets", command=zip_all_datasets, width=30, height=2).pack(pady=5)

    tk.Label(root, text="All outputs go to the 'datasets' folder", font=("Helvetica", 9)).pack(pady=15)

    root.mainloop()

if __name__ == "__main__":
    main()
