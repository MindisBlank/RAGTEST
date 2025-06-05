# File: src/process_raw.py

import os
from pathlib import Path

from extractors.pdf_extractor import extract_text_from_pdf
from extractors.docx_extractor import extract_text_from_docx
from extractors.pptx_extractor import extract_text_from_pptx

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

# Make sure the processed folder exists
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def process_file(input_path: Path, output_path: Path):
    ext = input_path.suffix.lower()
    if ext == ".pdf":
        text = extract_text_from_pdf(str(input_path))
    elif ext == ".docx":
        text = extract_text_from_docx(str(input_path))
    elif ext == ".pptx":
        text = extract_text_from_pptx(str(input_path))
    else:
        print(f"Skipping unsupported file type: {input_path.name}")
        return

    # Write extracted text to .txt in processed folder
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"→ Extracted {input_path.name} → {output_path.name}")

def main():
    for entry in RAW_DIR.iterdir():
        if entry.is_file():
            # Build output filename: same stem, but .txt
            out_file = PROCESSED_DIR / f"{entry.stem}.txt"
            process_file(entry, out_file)

if __name__ == "__main__":
    main()
