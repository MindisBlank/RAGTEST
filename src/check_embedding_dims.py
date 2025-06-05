# check_embedding_dims.py

import json
from pathlib import Path

EMBEDS_DIR = Path("data/embeddings")

dims = {}
for file in EMBEDS_DIR.glob("*.json"):
    try:
        data = json.loads(file.read_text(encoding="utf-8"))
        emb = data.get("embedding")
        if isinstance(emb, list):
            dims.setdefault(len(emb), 0)
            dims[len(emb)] += 1
        else:
            dims.setdefault("invalid", 0)
            dims["invalid"] += 1
    except Exception:
        dims.setdefault("parse_error", 0)
        dims["parse_error"] += 1

print("Embedding dimensions and counts:")
for dim, count in dims.items():
    print(f"  {dim}: {count} file(s)")
