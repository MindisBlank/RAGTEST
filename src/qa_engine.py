# src/qa_engine.py

import os
from typing import List, Dict
from google import genai
from dotenv import load_dotenv

from config import (
    GOOGLE_API_KEY,
    GOOGLE_GEMINI_MODEL,
)
from retriever import retrieve_top_k_chunks

load_dotenv()  # ensure GOOGLE_API_KEY, GOOGLE_GEMINI_MODEL are loaded

# Initialize Gemini client
if not GOOGLE_API_KEY:
    raise RuntimeError("Missing GOOGLE_API_KEY in environment or .env.")
gemini_client = genai.Client(api_key=GOOGLE_API_KEY)

def build_mini_prompt(
    chunks: List[Dict],
    user_question: str
) -> str:
    """
    Assemble the system instruction + retrieved chunks + user question.
    Each chunk is labeled 'Source: <chunk_id>'.
    """
    system_instruction = (
        "You are Veitur's on-demand internal-process expert. "
        "Below are excerpts from Veitur's official manuals and templates. "
        "Each excerpt is preceded by “Source: <chunk_id>”. "
        "Use only the information in these excerpts to answer questions about Veitur's processes. "
        "If no relevant information exists, reply: “I don't see relevant documentation.”"
    )

    # Format each chunk
    chunk_sections = []
    for chunk in chunks:
        chunk_id = chunk["id"]
        text = chunk["text"]
        chunk_sections.append(f"Source: {chunk_id}\n{text}")

    mini_prompt = (
        system_instruction
        + "\n\n"
        + "\n\n".join(chunk_sections)
        + "\n\n"
        + f"User question: {user_question}\n"
    )
    return mini_prompt

def answer_question(
    user_question: str,
    top_k: int = 5
) -> str:
    """
    1) Retrieve top_k chunks
    2) Build mini-prompt
    3) Call Gemini (or other LLM) to generate answer
    """
    # 1. Retrieve
    chunks = retrieve_top_k_chunks(user_question, k=top_k)

    # If no chunks or all scores are very low, optionally fallback
    if not chunks or chunks[0]["score"] < 0.15:
        return "I don't see relevant documentation about that topic in Veitur's corpus."

    # 2. Build prompt
    prompt = build_mini_prompt(chunks, user_question)

    # 3. Call Gemini
    try:
        response = gemini_client.models.generate_content(
            model=GOOGLE_GEMINI_MODEL,
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        raise RuntimeError(f"Gemini generation failed: {e}")

if __name__ == "__main__":
    import sys

    print("Veitur-AI Q&A (type 'exit' to quit)")
    while True:
        question = input("\nAsk a question: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        try:
            answer = answer_question(question, top_k=5)
            print(f"\n>> {answer}\n")
        except Exception as exc:
            print(f"[ERROR] {exc}")
