# Veitur RAG — Prototype 1

A first-pass retrieval-augmented generation (RAG) pipeline over Veitur's internal
engineering documentation. You point it at a folder of PDFs / Word / PowerPoint
files, it extracts the text, chunks it, embeds the chunks with Google's
`text-embedding-004`, stores the vectors in Pinecone, and answers questions with
Gemini using only the retrieved excerpts as context.

> **Status: prototype.** This is the first working version, built to prove the
> approach end to end — not a polished product. It is a set of scripts you run in
> order, not a packaged application. There are known rough edges; they're listed
> honestly under [Known issues](#known-issues) so nobody loses an afternoon to them.

The bundled corpus (`raw/`) is Icelandic technical documentation — distribution
substation handbooks, design guidelines for consultants, water-system control
guidelines. Questions and answers work in Icelandic.

---

## How it works

```
raw/*.pdf|docx|pptx
        |  process_raw.py      -> extractors/{pdf,docx,pptx}_extractor.py
        v
processed/*.txt                   plain text, one file per document
        |  process_chunks.py   -> extractors/chunker.py
        v
data/chunks/*.txt                 ~300 words each, 50-word overlap
        |  embedder.py         -> Google GenAI embed_content (768-dim)
        v
data/embeddings/*.json            { chunk_id, text, embedding }
        |  indexer.py          -> Pinecone upsert (cosine, serverless)
        v
Pinecone index "veitur-docs"
        |  retriever.py        -> embed question, top-K similarity search
        v
qa_engine.py                      builds a source-labelled prompt -> Gemini -> answer
```

The prompt in `qa_engine.py` instructs the model to answer **only** from the
retrieved excerpts, each labelled `Source: <chunk_id>`, and to say it has no
relevant documentation otherwise. There's also a score floor: if the best match
scores below `0.15`, the pipeline declines to answer rather than guessing.

---

## Repository layout

```
RAGTEST/
├── raw/                          # source documents (PDFs, committed)
├── processed/                    # extracted plain text (committed)
├── src/
│   ├── config.py                 # API keys, model names, data paths (reads .env)
│   ├── extractors/
│   │   ├── pdf_extractor.py      # PyPDF2 text extraction
│   │   ├── docx_extractor.py     # python-docx
│   │   ├── pptx_extractor.py     # python-pptx
│   │   └── chunker.py            # word-window chunking with overlap
│   ├── process_raw.py            # step 1: documents -> text
│   ├── process_chunks.py         # step 2: text -> chunks
│   ├── embedder.py               # step 3: chunks -> embedding JSON
│   ├── indexer.py                # step 4: embedding JSON -> Pinecone
│   ├── retriever.py              # top-K similarity search
│   ├── qa_engine.py              # prompt assembly + Gemini call + CLI loop
│   └── check_embedding_dims.py   # sanity check: vector dims across the JSON files
├── tests/
│   ├── test_extractors.py        # smoke script for the three extractors + chunker
│   ├── test_prompts.txt          # sample questions (incl. an off-corpus control)
│   └── sample_files/             # tiny test.pdf / test.docx / test.pptx
├── req.txt                       # pip dependencies
├── environment.yml.txt           # conda environment (rename to environment.yml)
└── LICENSE                       # MIT
```

`data/chunks/` and `data/embeddings/` are created on first run and are not
committed.

---

## Setup

**Requirements:** Python 3.9+ (3.10 is what this was built on), a Google AI
Studio API key, and a Pinecone account.

### 1. Environment

With conda:

```bash
mv environment.yml.txt environment.yml && conda env create -f environment.yml && conda activate veitur-rag
```

Or with pip:

```bash
python -m venv .venv && source .venv/bin/activate && pip install -r req.txt
```

Either way, install the Google client too — it's used by the code but missing
from both dependency files:

```bash
pip install google-genai
```

### 2. API keys

Create a `.env` file in the project root (`config.py` loads it via
`python-dotenv`):

```
GOOGLE_API_KEY=your-google-ai-studio-key
PINECONE_API_KEY=your-pinecone-key
PINECONE_ENV=us-east-1
GOOGLE_GEMINI_MODEL=gemini-2.0-flash
GOOGLE_EMBEDDING_MODEL=text-embedding-004
```

`.env` is not committed and must never be. The repo has no `.gitignore` yet, so
add one before you commit anything.

---

## Running the pipeline

Every script resolves its data paths **relative to the current working
directory** and imports its siblings as top-level modules, so run them from
inside `src/`:

```bash
cd src && python process_raw.py
```

```bash
cd src && python process_chunks.py
```

```bash
cd src && python embedder.py
```

```bash
cd src && python indexer.py
```

In order: documents to text, text to ~300-word chunks, chunks to embeddings
(already-embedded chunks are skipped, so re-running is cheap), embeddings into
Pinecone.

Then ask it things — this drops you into a prompt loop, `exit` or `quit` to
leave:

```bash
cd src && python qa_engine.py
```

`retriever.py` can also be run directly to inspect what comes back for a question
without spending a generation call:

```bash
cd src && python retriever.py
```

The extractor smoke test runs from the project root instead, since it imports via
the `src.` package path:

```bash
python tests/test_extractors.py
```

---

## Known issues

These are real and unfixed — prototype 1 warts, in rough order of how likely they
are to bite you:

- **Data paths don't match the repo layout.** `process_raw.py` reads `data/raw/`
  and writes `data/processed/`, but the committed folders are `raw/` and
  `processed/` at the top level. Until this is reconciled, either move the
  folders under `data/` or edit the `RAW_DIR` / `PROCESSED_DIR` constants. The
  chunk and embedding stages are self-consistent, so only step 1 is affected.
- **`google-genai` is missing from `req.txt` and `environment.yml.txt`,** while
  both still list `openai` and `pinecone-client`. The OpenAI settings in
  `config.py` are leftovers from the original plan and aren't used by any code
  path.
- **The embedding dimension is hardcoded to 768 in `indexer.py`.** There's a
  working `get_first_embedding_dimension()` helper right above it that goes
  uncalled — `main()` just assigns `dim = 768` inside a `try` that can't fail.
  Fine for `text-embedding-004`, wrong the moment you switch embedding models.
- **`indexer.py` will delete and recreate the index** without asking if the
  existing index's dimension differs from what it expects. Know that before you
  point it at an index you care about.
- **The default Gemini model name in `config.py` (`gemini-1.5-turbo`) isn't a
  real model.** Set `GOOGLE_GEMINI_MODEL` in `.env` and you'll never hit it.
- **`tests/test_extractors.py` isn't a real test suite** — it's a print-and-eyeball
  script with no assertions, and the fixtures it checks against were generated by
  code that is now commented out at the top of the file.
- **`__pycache__` directories are committed.** A `.gitignore` covering
  `__pycache__/`, `.env`, and `data/` would clean this up.
- Chunking splits on whitespace only, so it cuts across sentence and section
  boundaries. Retrieval quality on the Icelandic corpus is decent anyway, but
  structure-aware chunking is the obvious next improvement.

---

## License

MIT — see [LICENSE](LICENSE). Copyright (c) 2025 Jason Alexander Quinn.

The documents under `raw/` and `processed/` are Veitur materials and are not
covered by that license.
