# src/config.py

import os
from dotenv import load_dotenv

load_dotenv()  # looks for a .env file in the project root


# ─── Google Gemini settings ──────────────────────────────────────
GOOGLE_API_KEY           = os.getenv("GOOGLE_API_KEY")
# Pick the Gemini model you want (e.g. "gemini-1.5-turbo" or "gemini-2.0-flash").
# You can check the exact name in Google AI Studio or the GenAI docs.
GOOGLE_GEMINI_MODEL      = os.getenv("GOOGLE_GEMINI_MODEL", "gemini-1.5-turbo")
GOOGLE_EMBEDDING_MODEL    = os.getenv("GOOGLE_EMBEDDING_MODEL", "text-embedding-004")

# ─── OpenAI settings ───────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Choose your embedding model; "text-embedding-ada-002" is common:
OPENAI_EMBEDDING_MODEL = "text-embedding-ada-002"

# ─── Pinecone settings ─────────────────────────────────
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV      = os.getenv("PINECONE_ENV", "us-east-1")  # e.g. "us-west1-gcp" or your preferred region
# The name of the index where we’ll store embeddings; pick something unique:
PINECONE_INDEX_NAME = "veitur-docs"

# ─── Data folder paths ──────────────────────────────────
CHUNKS_DIR   = "data/chunks"       # where chunked .txt files live
EMBEDS_DIR   = "data/embeddings"   # where to locally save embeddings (optional)