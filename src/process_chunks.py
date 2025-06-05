# File: src/process_chunks.py

import os
from pathlib import Path

from extractors.chunker import chunk_text

RAW_TXT_DIR = Path("data/processed")
CHUNKS_DIR   = Path("data/chunks")

# Make sure data/chunks/ exists
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

# Choose roughly how many words per chunk and how much overlap
WORD_CHUNK_SIZE = 300  # adjust if needed
WORD_OVERLAP    = 50   # number of words to overlap between chunks

def process_txt_file(input_path: Path):
    """
    Read the full text from input_path, split it into chunks of ~WORD_CHUNK_SIZE words
    with WORD_OVERLAP words of overlap, and write each chunk out as a separate .txt file
    under data/chunks/.
    """
    text = input_path.read_text(encoding="utf-8")
    # Pass both chunk_size and overlap to the updated chunk_text function
    chunks = chunk_text(text, chunk_size=WORD_CHUNK_SIZE, overlap=WORD_OVERLAP)

    base_name = input_path.stem  # e.g. "OnboardingGuide"
    for idx, chunk in enumerate(chunks, start=1):
        chunk_filename = f"{base_name}_chunk_{idx}.txt"
        out_path = CHUNKS_DIR / chunk_filename
        out_path.write_text(chunk, encoding="utf-8")
    print(f"→ {input_path.name}: split into {len(chunks)} chunks.")

def main():
    for txt_file in RAW_TXT_DIR.glob("*.txt"):
        process_txt_file(txt_file)

if __name__ == "__main__":
    main()
