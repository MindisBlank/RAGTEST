from docx import Document

def extract_text_from_docx(input_path: str) -> str:
    """
    Extracts text from a .docx file and returns it as a single string.
    """
    doc = Document(input_path)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text
