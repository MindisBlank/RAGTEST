import os
from src.extractors.pdf_extractor import extract_text_from_pdf
from src.extractors.docx_extractor import extract_text_from_docx
from src.extractors.pptx_extractor import extract_text_from_pptx
from src.extractors.chunker import chunk_text

# # Create sample files for testing
# os.makedirs("tests/sample_files", exist_ok=True)

# # 1) Create a simple DOCX
# from docx import Document as DocxDocument
# doc = DocxDocument()
# doc.add_paragraph("Hello, this is a test DOCX file.")
# doc.save("tests/sample_files/test.docx")

# # 2) Create a simple PPTX
# from pptx import Presentation as PptxPresentation
# prs = PptxPresentation()
# slide_layout = prs.slide_layouts[1]  # Title and Content layout
# slide = prs.slides.add_slide(slide_layout)
# title = slide.shapes.title
# body = slide.shapes.placeholders[1]
# title.text = "Test Slide"
# body.text = "This is a test PPTX file."
# prs.save("tests/sample_files/test.pptx")

# # 3) Create a blank PDF (no text)
# from PyPDF2 import PdfWriter
# writer = PdfWriter()
# writer.add_blank_page(width=72, height=72)
# with open("tests/sample_files/test.pdf", "wb") as f:
#     writer.write(f)

# Test DOCX extractor
docx_text = extract_text_from_docx("tests/sample_files/test.docx")
print("DOCX Extracted Text:")
print(docx_text)

# Test PPTX extractor
pptx_text = extract_text_from_pptx("tests/sample_files/test.pptx")
print("PPTX Extracted Text:")
print(pptx_text)

# Test PDF extractor (should be empty)
pdf_text = extract_text_from_pdf("tests/sample_files/test.pdf")
print("PDF Extracted Text (expected empty or whitespace):")
print(repr(pdf_text))

# Test chunker
sample = "This is a sample text " * 50  # ~250 words
chunks = chunk_text(sample, chunk_size=50)
print("Number of chunks (expected ~5):", len(chunks))
for i, chunk in enumerate(chunks, 1):
    print(f"Chunk {i} word count:", len(chunk.split()))
