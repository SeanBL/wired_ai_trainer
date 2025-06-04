import os
import json
from colorama import init, Fore, Style

init(autoreset=True)

def combine_sbert_datasets(folder_path, output_path):
    if not os.path.isdir(folder_path):
        print(Fore.RED + f"❌ Folder not found: {folder_path}")
        return

    combined_data = []
    files = [f for f in os.listdir(folder_path) if f.endswith(".json")]

    if not files:
        print(Fore.YELLOW + "⚠️ No SBERT dataset files found to combine.")
        return

    for file in files:
        full_path = os.path.join(folder_path, file)
        with open(full_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                combined_data.extend(data)
                print(Fore.GREEN + f"✅ Loaded: {file} ({len(data)} pairs)")
            else:
                print(Fore.YELLOW + f"⚠️ Skipped (not a list): {file}")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, indent=2, ensure_ascii=False)

    print(Style.BRIGHT + f"\n🎉 Combined SBERT dataset saved to: {output_path}")
    print(Fore.CYAN + f"📊 Total sentence pairs: {len(combined_data)}")


if __name__ == "__main__":
    sbert_folder = os.path.join("datasets", "sbert")
    output_file = os.path.join("datasets", "sbert_combined.json")
    combine_sbert_datasets(sbert_folder, output_file)
