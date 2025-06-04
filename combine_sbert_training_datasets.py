import os
import json
from colorama import init, Fore

init(autoreset=True)

def combine_sbert_training_datasets(folder_path, output_path):
    combined_data = []
    files = [f for f in os.listdir(folder_path) if f.endswith(".json")]

    if not files:
        print(Fore.RED + "❌ No SBERT training dataset files found.")
        return

    print(Fore.CYAN + f"📁 Combining {len(files)} files from '{folder_path}'...\n")

    for file in files:
        full_path = os.path.join(folder_path, file)
        with open(full_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    combined_data.extend(data)
                    print(Fore.GREEN + f"✅ Loaded {file} ({len(data)} pairs)")
                else:
                    print(Fore.YELLOW + f"⚠️ Skipped '{file}' (invalid format)")
            except Exception as e:
                print(Fore.RED + f"❌ Error loading '{file}': {e}")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, indent=2, ensure_ascii=False)

    print(Fore.CYAN + f"\n🎉 Combined training data saved to: {output_path}")
    print(Fore.BLUE + f"📊 Total training pairs: {len(combined_data)}")


if __name__ == "__main__":
    folder = "datasets/sbert_train"
    output = "datasets/sbert_combined_train.json"
    combine_sbert_training_datasets(folder, output)
