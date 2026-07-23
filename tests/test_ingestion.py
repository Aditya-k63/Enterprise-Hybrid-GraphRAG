from app.ingestion.chunker import chunk_text
from app.ingestion.pdf_parser import extract_text_from_pdf


def test_chunk_text():
    text = "This is a test sentence. " * 100
    chunks = chunk_text(text)
    assert len(chunks) > 0
    assert all(isinstance(c, str) for c in chunks)
    assert all(len(c) <= 700 for c in chunks)


def test_chunk_text_empty():
    chunks = chunk_text("")
    assert chunks == []


def test_extract_text_from_pdf():
    from io import BytesIO
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    buf = BytesIO()
    writer.write(buf)
    buf.seek(0)

    text = extract_text_from_pdf(buf.read())
    assert isinstance(text, str)
