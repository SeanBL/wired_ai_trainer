import os
import json

def validate_json_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if "data" not in data:
            return False, "Missing 'data' key."

        for entry in data["data"]:
            if "title" not in entry or "paragraphs" not in entry:
                return False, "Missing 'title' or 'paragraphs' inside 'data'."
            for para in entry["paragraphs"]:
                if "context" not in para or "qas" not in para:
                    return False, "Missing 'context' or 'qas' inside 'paragraph'."

        return True, "Valid."
    except Exception as e:
        return False, f"Exception: {str(e)}"

def main():
    datasets_folder = "datasets"
    if not os.path.exists(datasets_folder):
        print(f"❌ Folder '{datasets_folder}' not found.")
        return

    json_files = [f for f in os.listdir(datasets_folder) if f.endswith(".json")]
    if not json_files:
        print(f"❌ No JSON files found in '{datasets_folder}'.")
        return

    print(f"🔍 Validating {len(json_files)} dataset files...\n")

    for json_file in json_files:
        filepath = os.path.join(datasets_folder, json_file)
        valid, message = validate_json_file(filepath)
        if valid:
            print(f"✅ {json_file} — {message}")
        else:
            print(f"❌ {json_file} — {message}")

    print("\n🎯 Validation complete.")

if __name__ == "__main__":
    main()
