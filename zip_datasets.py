import os
import zipfile

def zip_datasets_folder():
    datasets_folder = "datasets"
    output_zip = "datasets.zip"

    if not os.path.exists(datasets_folder):
        print(f"❌ Folder '{datasets_folder}' not found.")
        return

    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(datasets_folder):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, start=datasets_folder)
                zipf.write(file_path, arcname)

    print(f"✅ Created '{output_zip}' with all datasets.")

def main():
    zip_datasets_folder()

if __name__ == "__main__":
    main()
