# src/indexer.py

import os
import json
import unicodedata
import re
from pathlib import Path

from pinecone import Pinecone, ServerlessSpec
from pinecone.exceptions import PineconeException

from config import (
    PINECONE_API_KEY,
    PINECONE_ENV,
    PINECONE_INDEX_NAME,
    EMBEDS_DIR,
)

# ─── Initialize the Pinecone client ────────────────────────────────────
if not PINECONE_API_KEY:
    raise RuntimeError("Missing PINECONE_API_KEY in environment or .env.")

pc = Pinecone(api_key=PINECONE_API_KEY)
REGION = PINECONE_ENV  # e.g., "us-east-1"

def sanitize_id(original_id: str) -> str:
    """
    Normalize Unicode to NFKD, drop non-ASCII, then replace any
    remaining characters except [A-Za-z0-9\-_] with underscores.
    """
    # 1. Decompose accents (NFKD) and drop non-ASCII bytes
    nfkd = unicodedata.normalize("NFKD", original_id)
    ascii_only = nfkd.encode("ASCII", "ignore").decode("ASCII")
    # 2. Replace anything that is not a letter, digit, hyphen, or underscore with '_'
    sanitized = re.sub(r"[^A-Za-z0-9\-_]", "_", ascii_only)
    # 3. Avoid leading/trailing underscores (optional)
    sanitized = sanitized.strip("_")
    # 4. If it becomes blank for some reason, fall back to a hex digest of original_id
    if len(sanitized) == 0:
        import hashlib
        return hashlib.sha256(original_id.encode("utf-8")).hexdigest()[:32]
    return sanitized

def get_first_embedding_dimension() -> int:
    """
    Scan EMBEDS_DIR for the first valid JSON file and return the length
    of its 'embedding' vector. Raises if no valid file is found.
    """
    embed_path = Path(EMBEDS_DIR)
    if not embed_path.exists():
        raise RuntimeError(f"Embeddings folder not found: {EMBEDS_DIR}")

    for json_file in embed_path.glob("*.json"):
        try:
            record = json.loads(json_file.read_text(encoding="utf-8"))
            embedding = record.get("embedding")
            if isinstance(embedding, list) and len(embedding) > 0:
                return len(embedding)
        except json.JSONDecodeError:
            # Skip invalid JSON, try next file
            continue

    raise RuntimeError(f"No valid embedding JSON found in {EMBEDS_DIR}.")

def ensure_index_exists(index_name: str, dimension: int):
    """
    If the index does not exist, create it with the given dimension.
    If it exists but has a different dimension, delete + recreate it.
    """
    existing = pc.list_indexes().names()

    if index_name in existing:
        # Fetch existing index info to check dimension
        try:
            info = pc.describe_index(index_name)
            existing_dim = info["dimension"]
        except Exception as e:
            raise RuntimeError(f"Unable to describe existing index '{index_name}': {e}")

        if existing_dim == dimension:
            print(f"Index '{index_name}' already exists with dimension {dimension}. Skipping creation.")
            return
        else:
            print(
                f"Index '{index_name}' exists with dimension {existing_dim}, "
                f"but we need dimension {dimension}. Deleting and recreating…"
            )
            try:
                pc.delete_index(index_name)
                print(f"✅ Deleted index '{index_name}'.")
            except PineconeException as e:
                raise RuntimeError(f"Failed to delete index '{index_name}': {e}")

    # (Re)create the index
    print(f"Creating Pinecone index '{index_name}' with dimension {dimension} in region {REGION}…")
    try:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=REGION),
        )
        print(f"✅ Created Pinecone index: '{index_name}' (region={REGION}, dim={dimension})")
    except PineconeException as e:
        raise RuntimeError(f"Failed to create index '{index_name}': {e}") from e

def main():
    # 1. Determine embedding dimension from first valid JSON
    try:
        dim = 768
        print(f"Detected embedding dimension: {dim}")
    except RuntimeError as e:
        raise RuntimeError(f"Cannot determine embedding dimension: {e}")

    # 2. Ensure Pinecone index exists (or recreate if dimension mismatch)
    ensure_index_exists(PINECONE_INDEX_NAME, dim)

    # 3. Get a handle to the index
    try:
        index = pc.Index(PINECONE_INDEX_NAME)
    except PineconeException as e:
        raise RuntimeError(f"Failed to connect to index '{PINECONE_INDEX_NAME}': {e}")

    # 4. Collect all embedding JSON files
    embed_path = Path(EMBEDS_DIR)
    json_files = list(embed_path.glob("*.json"))
    if not json_files:
        print(f"No JSON embedding files found in {EMBEDS_DIR}. Nothing to index.")
        return

    print(f"Found {len(json_files)} embedding files. Starting upsert…")

    batch_size = 32
    buffer = []
    total_upserted = 0
    total_batches = 0

    for idx, json_file in enumerate(json_files, start=1):
        try:
            record = json.loads(json_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            print(f"[ERROR] Invalid JSON in '{json_file.name}': {e}. Skipping.")
            continue

        original_id = record.get("chunk_id")
        embedding = record.get("embedding")
        text = record.get("text", "")

        if original_id is None or embedding is None:
            print(f"[WARNING] Missing 'chunk_id' or 'embedding' in '{json_file.name}'. Skipping.")
            continue

        # Verify embedding length matches expected dimension
        if not isinstance(embedding, list) or len(embedding) != dim:
            print(
                f"[ERROR] Dimension mismatch in '{json_file.name}': "
                f"expected {dim}, got {len(embedding)}. Skipping."
            )
            continue

        # 5. Sanitize the chunk_id into pure ASCII
        sanitized = sanitize_id(original_id)

        # Store the original_id in metadata for traceability, but use sanitized for Pinecone
        metadata = {"original_chunk_id": original_id, "text": text}
        buffer.append((sanitized, embedding, metadata))

        # Upsert once buffer reaches batch_size
        if len(buffer) >= batch_size:
            try:
                index.upsert(vectors=buffer)
                total_upserted += len(buffer)
                total_batches += 1
                print(f"✅ Upserted batch {total_batches}: {len(buffer)} vectors (up to file #{idx}).")
            except PineconeException as e:
                print(f"[ERROR] Failed to upsert batch ending at file #{idx}: {e}")
            finally:
                buffer.clear()

    # Upsert any remaining vectors
    if buffer:
        try:
            index.upsert(vectors=buffer)
            total_upserted += len(buffer)
            total_batches += 1
            print(f"✅ Upserted final batch {total_batches}: {len(buffer)} vectors.")
        except PineconeException as e:
            print(f"[ERROR] Failed to upsert final batch: {e}")
        buffer.clear()

    print(f"Upsert complete: {total_upserted} total vectors upserted in {total_batches} batches.")

    # 6. Print index statistics
    try:
        stats = index.describe_index_stats()
        vector_count = stats.get("total_vector_count", "unknown")
        print(f"Index '{PINECONE_INDEX_NAME}' now contains {vector_count} vectors.")
    except PineconeException as e:
        print(f"[WARNING] Could not fetch index stats: {e}")

if __name__ == "__main__":
    main()
