def load_pdf(file_path: str) -> str:
    """Extract all available text from a PDF as a single string."""

    pages = load_pdf_pages(file_path)
    return "\n".join(page["text"] for page in pages if page["text"])


def load_pdf_pages(file_path: str) -> list[dict]:
    """Extract PDF text page-by-page with page metadata."""

    from pypdf import PdfReader

    reader = PdfReader(file_path)
    page_payloads: list[dict] = []

    for idx, page in enumerate(reader.pages, start=1):
        extracted = page.extract_text() or ""
        page_payloads.append({"page": idx, "text": extracted.strip()})

    return page_payloads
