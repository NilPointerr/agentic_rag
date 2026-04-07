import shutil
import subprocess
import tempfile
from pathlib import Path

from pypdf import PdfReader


class PdfTextExtractionError(Exception):
    pass


class ImageOnlyPdfError(PdfTextExtractionError):
    pass


def load_pdf(file_path: str) -> str:
    """Load all extracted page text from a PDF into a single string."""
    pages = load_pdf_pages(file_path)
    return "\n".join(page["text"] for page in pages if page["text"])


def load_pdf_pages(file_path: str) -> list[dict]:
    """Extract text page by page from a PDF, with optional OCR fallback."""
    reader = PdfReader(file_path)
    pages: list[dict] = []

    for page_number, page in enumerate(reader.pages, start=1):
        extracted = page.extract_text() or ""
        pages.append(
            {
                "page_number": page_number,
                "text": extracted.strip(),
            }
        )

    if any(page["text"] for page in pages):
        return pages

    if _has_command("tesseract") and _has_command("pdftoppm"):
        ocr_pages = _load_pdf_pages_with_tesseract(file_path, page_count=len(pages))
        if any(page["text"] for page in ocr_pages):
            return ocr_pages

    raise ImageOnlyPdfError(
        "This PDF appears to be image-only or scanned. No selectable text was found.pls parse text selectable pdf as of now ,we will support image with pdf shortly "
    )


def _has_command(command: str) -> bool:
    """Return whether a shell command is available on the host."""
    return shutil.which(command) is not None


def _render_pdf_to_images(file_path: str, output_prefix: Path) -> None:
    """Render PDF pages to PNG images for OCR processing."""
    subprocess.run(
        [
            "pdftoppm",
            "-png",
            file_path,
            str(output_prefix),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def _load_pdf_pages_with_tesseract(file_path: str, page_count: int) -> list[dict]:
    """Use Tesseract OCR to extract text from rendered PDF page images."""
    if not (_has_command("tesseract") and _has_command("pdftoppm")):
        return []

    pages: list[dict] = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_prefix = Path(tmp_dir) / "page"
        _render_pdf_to_images(file_path, output_prefix)

        for page_number in range(1, page_count + 1):
            image_path = Path(f"{output_prefix}-{page_number}.png")
            extracted = subprocess.run(
                [
                    "tesseract",
                    str(image_path),
                    "stdout",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            pages.append(
                {
                    "page_number": page_number,
                    "text": extracted.stdout.strip(),
                }
            )

    return pages
