veitur_AI/
├── data/
│   ├── raw/                  # Original PDFs, DOCXs, PPTXs (uploaded by you)
│   ├── processed/            # Text chunks extracted from those files
│   └── embeddings/           # Optional: store serialized embeddings if needed
│
├── src/
│   ├── extractors/
│   │   ├── pdf_extractor.py      # Extract text from PDFs
│   │   ├── docx_extractor.py     # Extract text from Word files
│   │   ├── pptx_extractor.py     # Extract text from PowerPoints
│   │   └── chunker.py            # Break text into small chunks (e.g., 200-300 words)
│   │
│   ├── embedder.py               # Uses OpenAI or other API to generate embeddings
│   ├── indexer.py                # Pushes embeddings to Pinecone or other vector DB
│   ├── retriever.py              # Runs similarity search and gets top-K chunks
│   ├── qa_engine.py              # Builds the prompt and sends it to ChatGPT
│   └── config.py                 # Central place for keys, model names, etc.
│
├── notebooks/
│   └── dev-testing.ipynb        # For manual testing of queries and answers
│
├── app.py                       # CLI or web entry point (Streamlit, Flask, etc.)
├── environment.yml              # If using conda (includes Python version)
├── req.txt                      # If using pip
└── README.md                    # Project overview and setup instructions
