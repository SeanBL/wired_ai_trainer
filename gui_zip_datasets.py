import os
import zipfile
import tkinter as tk
from tkinter import messagebox

def zip_datasets():
    folder = "datasets"
    if not os.path.exists(folder):
        messagebox.showerror("Error", f"No '{folder}' folder found.")
        return

    json_files = [f for f in os.listdir(folder) if f.endswith(".json")]
    if not json_files:
        messagebox.showerror("Error", "No .json files to zip.")
        return

    with zipfile.ZipFile("datasets.zip", 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in json_files:
            filepath = os.path.join(folder, file)
            zipf.write(filepath, arcname=file)

    messagebox.showinfo("Success", "✅ Datasets zipped into 'datasets.zip'.")

def main():
    root = tk.Tk()
    root.title("Zip Datasets")

    canvas = tk.Canvas(root, width=300, height=150)
    canvas.pack()

    button = tk.Button(root, text="Zip All Datasets", command=zip_datasets, height=2, width=30)
    canvas.create_window(150, 75, window=button)

    root.mainloop()

if __name__ == "__main__":
    main()
