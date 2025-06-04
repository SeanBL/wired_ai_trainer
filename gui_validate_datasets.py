import os
import json
import tkinter as tk
from tkinter import messagebox

def validate_json_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if "data" not in data:
            return False, "Missing 'data' key."

        for entry in data["data"]:
            if "title" not in entry or "paragraphs" not in entry:
                return False, "Missing 'title' or 'paragraphs'."
            for para in entry["paragraphs"]:
                if "context" not in para or "qas" not in para:
                    return False, "Missing 'context' or 'qas'."

        return True, "Valid."
    except Exception as e:
        return False, f"Invalid JSON: {str(e)}"

def validate_all_datasets():
    folder = "datasets"
    if not os.path.exists(folder):
        messagebox.showerror("Error", f"No '{folder}' folder found.")
        return

    json_files = [f for f in os.listdir(folder) if f.endswith(".json")]
    if not json_files:
        messagebox.showerror("Error", "No .json files found in 'datasets' folder.")
        return

    report = []
    for file in json_files:
        filepath = os.path.join(folder, file)
        valid, message = validate_json_file(filepath)
        if valid:
            report.append(f"✅ {file}: {message}")
        else:
            report.append(f"❌ {file}: {message}")

    summary = "\n".join(report)
    with open(os.path.join(folder, "validation_report.txt"), "w", encoding="utf-8") as f:
        f.write(summary)

    messagebox.showinfo("Validation Done", f"✅ Validation complete!\nSee 'validation_report.txt' in datasets.")

def main():
    root = tk.Tk()
    root.title("Validate Datasets")

    canvas = tk.Canvas(root, width=300, height=150)
    canvas.pack()

    button = tk.Button(root, text="Validate All Datasets", command=validate_all_datasets, height=2, width=30)
    canvas.create_window(150, 75, window=button)

    root.mainloop()

if __name__ == "__main__":
    main()
