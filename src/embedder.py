# src/embedding.py

import os
import json
from pathlib import Path
from time import sleep

from google import genai

from config import (
    GOOGLE_API_KEY,
    GOOGLE_EMBEDDING_MODEL,
    CHUNKS_DIR,
    EMBEDS_DIR,
)

# ─── Initialize the GenAI Client ────────────────────────────────────
if not GOOGLE_API_KEY:
    raise RuntimeError("Missing GOOGLE_API_KEY in environment or .env.")

# Create a GenAI client with your API key
client = genai.Client(api_key=GOOGLE_API_KEY)

# Ensure the embeddings folder exists
Path(EMBEDS_DIR).mkdir(parents=True, exist_ok=True)


def get_embedding(text: str) -> list[float]:
    """
    Call Google GenAI's embed_content endpoint to embed a single chunk of text.
    Retries up to 3 times on transient errors, then returns a plain list of floats.
    """
    for attempt in range(1, 4):
        try:
            # embed_content expects a string or list of strings. We pass one string.
            response = client.models.embed_content(
                model=GOOGLE_EMBEDDING_MODEL,
                contents=text
            )
            # response.embeddings is a list of ContentEmbedding objects.
            # Each ContentEmbedding has a `values` attribute (the raw float list).
            raw = response.embeddings[0]
            # Convert to a plain Python list of floats:
            return list(raw.values)
        except Exception as e:
            # Print the exception so you can see if it's a rate-limit or something else
            print(f"[Attempt {attempt}] Error calling embed_content: {e!r}")
            # Simple exponential backoff
            sleep(1 + attempt * 2)

    # If all 3 attempts failed:
    raise RuntimeError("Google embedding failed after 3 retries.")


def main():
    chunks_path = Path(CHUNKS_DIR)
    embed_path  = Path(EMBEDS_DIR)

    # Iterate over every .txt chunk file
    for txt_file in chunks_path.glob("*.txt"):
        base_name = txt_file.stem  # e.g. "OnboardingGuide_chunk_1"
        out_file  = embed_path / f"{base_name}.json"

        # Skip if already embedded
        if out_file.exists():
            print(f"Skipping {base_name} (already embedded).")
            continue

        text = txt_file.read_text(encoding="utf-8")
        print(f"Embedding chunk: {base_name}…")

        embedding_list = get_embedding(text)

        # Build a small JSON record
        record = {
            "chunk_id": base_name,
            "text":     text,
            "embedding": embedding_list,
        }

        # Write it out to data/embeddings/<chunk_id>.json
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(record, f)

        print(f"✅ Embedded {base_name} → {out_file.name}")

    print("All chunks embedded via Google-genai embed_content.")


if __name__ == "__main__":
    main()
