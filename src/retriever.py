# src/retriever.py

import os
from typing import List, Dict

from pinecone import Pinecone
from google import genai

from config import (
    GOOGLE_API_KEY,
    GOOGLE_EMBEDDING_MODEL,
    PINECONE_API_KEY,
    PINECONE_ENV,
    PINECONE_INDEX_NAME,
)

# ─── Initialize GenAI for embeddings ────────────────────────────
if not GOOGLE_API_KEY:
    raise RuntimeError("Missing GOOGLE_API_KEY in environment or .env.")

genai_client = genai.Client(api_key=GOOGLE_API_KEY)

# ─── Initialize Pinecone ────────────────────────────────────────
if not PINECONE_API_KEY:
    raise RuntimeError("Missing PINECONE_API_KEY in environment or .env.")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

def embed_query(text: str) -> List[float]:
    """
    Embed a single query string using Text Embedding 004 (via google-genai).
    Returns a 768-length list of floats.
    """
    response = genai_client.models.embed_content(
        model=GOOGLE_EMBEDDING_MODEL,
        contents=text
    )
    embedding_obj = response.embeddings[0]
    return list(embedding_obj.values)

def retrieve_top_k_chunks(
    user_question: str,
    k: int = 5
) -> List[Dict]:
    """
    1) Embed `user_question` into a vector.
    2) Query Pinecone for the top-K similar chunks (single-vector API).
    3) Return a list of dicts containing: { "id", "score", "text" }.
    """
    # 1) Embed the question
    try:
        q_vector = embed_query(user_question)
    except Exception as e:
        raise RuntimeError(f"Failed to embed query: {e}")

    # 2) Query Pinecone using the `vector=` parameter
    try:
        response = index.query(
            vector=q_vector,
            top_k=k,
            include_metadata=True
        )
    except Exception as e:
        raise RuntimeError(f"Pinecone query failed: {e}")

    # 3) Parse the matches from response["matches"]
    matches = response.get("matches", [])
    results = []
    for match in matches:
        chunk_id = match["id"]
        score = match["score"]
        metadata = match.get("metadata", {})
        chunk_text = metadata.get("text", "")
        results.append({
            "id": chunk_id,
            "score": score,
            "text": chunk_text
        })

    return results

if __name__ == "__main__":
    # Quick sanity test
    test_question = "Hvaða teikningar eru í forhönnunarsett"
    try:
        top_chunks = retrieve_top_k_chunks(test_question, k=3)
    except Exception as e:
        print(f"[ERROR] {e}")
    else:
        print(f"Top {len(top_chunks)} chunks for: “{test_question}”\n")
        for i, chunk in enumerate(top_chunks, 1):
            print(f"#{i} ID: {chunk['id']} (score: {chunk['score']:.4f})")
            snippet = chunk["text"][:200].replace("\n", " ") + "…"
            print(f"    {snippet}\n")
