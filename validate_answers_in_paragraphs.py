import os
import difflib
import string

def normalize(text):
    return (
        text.lower()
        .strip()
        .strip(string.punctuation)
        .replace("’", "'")
        .replace("“", '"')
        .replace("”", '"')
    )

def is_fuzzy_match(answer, paragraph, threshold=0.95):
    answer = normalize(answer)
    paragraph = normalize(paragraph)
    ratio = difflib.SequenceMatcher(None, answer, paragraph).ratio()
    return ratio >= threshold

def validate_module(input_file, threshold=0.95):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()

    title = ""
    paragraphs = []
    current_paragraph = ""
    current_qas = []

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

    # Add last paragraph
    if current_paragraph:
        paragraphs.append((current_paragraph.strip(), current_qas))

    print(f"\n🔍 Validating answers in: {input_file}")
    total = 0
    skipped = 0

    for idx, (paragraph, qas) in enumerate(paragraphs):
        for question, answer in qas:
            total += 1
            if normalize(answer) not in normalize(paragraph) and not is_fuzzy_match(answer, paragraph, threshold):
                skipped += 1
                print(f"⚠️ Skipped (not matched):\n  Q: {question}\n  A: {answer}\n  🔎 In Paragraph {idx + 1}\n")

    print(f"\n📊 Validation Complete:")
    print(f"✅ Answers found or fuzzy-matched: {total - skipped}")
    print(f"❌ Answers skipped: {skipped}")
    print(f"📄 Total QAs checked: {total}")


if __name__ == "__main__":
    validate_module("txt_files/FA_cuts_scrapes.txt")


