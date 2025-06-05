import PyPDF2

def extract_text_from_pdf(input_path: str) -> str:
    """
    Extracts text from a PDF file and returns it as a single string.
    """
    text = ""
    with open(input_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text
