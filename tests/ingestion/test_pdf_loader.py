from types import SimpleNamespace

import pytest

from app.ingestion.pdf_loader import (
    ImageOnlyPdfError,
    _has_command,
    _load_pdf_pages_with_tesseract,
    _render_pdf_to_images,
    load_pdf,
    load_pdf_pages,
)


def test_load_pdf_joins_non_empty_page_text(monkeypatch):
    """Full-document loading should concatenate page text with newlines."""
    monkeypatch.setattr(
        "app.ingestion.pdf_loader.load_pdf_pages",
        lambda path: [
            {"page_number": 1, "text": "alpha"},
            {"page_number": 2, "text": ""},
            {"page_number": 3, "text": "beta"},
        ],
    )

    assert load_pdf("guide.pdf") == "alpha\nbeta"


def test_load_pdf_pages_returns_extracted_text_without_ocr(monkeypatch):
    """Selectable text should be returned directly when any page has content."""

    class FakeReader:
        def __init__(self, file_path):
            self.pages = [
                SimpleNamespace(extract_text=lambda: "First page"),
                SimpleNamespace(extract_text=lambda: "  "),
            ]

    monkeypatch.setattr("app.ingestion.pdf_loader.PdfReader", FakeReader)

    pages = load_pdf_pages("guide.pdf")

    assert pages == [
        {"page_number": 1, "text": "First page"},
        {"page_number": 2, "text": ""},
    ]


def test_load_pdf_pages_uses_ocr_when_commands_available(monkeypatch):
    """OCR should be used for image-only PDFs when required tools exist."""

    class FakeReader:
        def __init__(self, file_path):
            self.pages = [
                SimpleNamespace(extract_text=lambda: ""),
                SimpleNamespace(extract_text=lambda: None),
            ]

    monkeypatch.setattr("app.ingestion.pdf_loader.PdfReader", FakeReader)
    monkeypatch.setattr("app.ingestion.pdf_loader._has_command", lambda command: True)
    monkeypatch.setattr(
        "app.ingestion.pdf_loader._load_pdf_pages_with_tesseract",
        lambda file_path, page_count: [
            {"page_number": 1, "text": "OCR page 1"},
            {"page_number": 2, "text": ""},
        ],
    )

    pages = load_pdf_pages("scan.pdf")

    assert pages[0]["text"] == "OCR page 1"


def test_load_pdf_pages_raises_for_image_only_pdf_without_ocr(monkeypatch):
    """Image-only PDFs should raise a clear error when OCR is unavailable."""

    class FakeReader:
        def __init__(self, file_path):
            self.pages = [SimpleNamespace(extract_text=lambda: "")]

    monkeypatch.setattr("app.ingestion.pdf_loader.PdfReader", FakeReader)
    monkeypatch.setattr("app.ingestion.pdf_loader._has_command", lambda command: False)

    with pytest.raises(ImageOnlyPdfError):
        load_pdf_pages("scan.pdf")


def test_has_command_uses_shutil_which(monkeypatch):
    """Command detection should delegate to shutil.which."""
    monkeypatch.setattr("app.ingestion.pdf_loader.shutil.which", lambda command: "/usr/bin/tool")
    assert _has_command("tool") is True

    monkeypatch.setattr("app.ingestion.pdf_loader.shutil.which", lambda command: None)
    assert _has_command("tool") is False


def test_render_pdf_to_images_invokes_pdftoppm(monkeypatch, tmp_path):
    """PDF rendering should shell out to pdftoppm with PNG output enabled."""
    captured = {}

    def fake_run(cmd, check, capture_output, text):
        captured["cmd"] = cmd
        return SimpleNamespace()

    monkeypatch.setattr("app.ingestion.pdf_loader.subprocess.run", fake_run)

    _render_pdf_to_images("guide.pdf", tmp_path / "page")

    assert captured["cmd"] == ["pdftoppm", "-png", "guide.pdf", str(tmp_path / "page")]


def test_load_pdf_pages_with_tesseract_returns_empty_when_commands_missing(monkeypatch):
    """OCR helper should short-circuit when host tools are unavailable."""
    monkeypatch.setattr("app.ingestion.pdf_loader._has_command", lambda command: False)

    assert _load_pdf_pages_with_tesseract("scan.pdf", page_count=2) == []


def test_load_pdf_pages_with_tesseract_extracts_each_page(monkeypatch):
    """OCR helper should render pages and collect Tesseract stdout per page."""
    monkeypatch.setattr("app.ingestion.pdf_loader._has_command", lambda command: True)
    monkeypatch.setattr("app.ingestion.pdf_loader._render_pdf_to_images", lambda *args: None)

    def fake_run(cmd, check, capture_output, text):
        page_marker = cmd[1]
        page_number = 1 if page_marker.endswith("-1.png") else 2
        return SimpleNamespace(stdout=f"ocr text {page_number}\n")

    monkeypatch.setattr("app.ingestion.pdf_loader.subprocess.run", fake_run)

    pages = _load_pdf_pages_with_tesseract("scan.pdf", page_count=2)

    assert pages == [
        {"page_number": 1, "text": "ocr text 1"},
        {"page_number": 2, "text": "ocr text 2"},
    ]
