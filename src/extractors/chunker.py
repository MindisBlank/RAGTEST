def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """
    Splits the input text into chunks of `chunk_size` words, with `overlap` words 
    shared between consecutive chunks.
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        # Move the window forward by chunk_size - overlap
        start += (chunk_size - overlap)

    return chunks
