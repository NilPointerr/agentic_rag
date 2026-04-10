from app.ingestion.chunker import chunk_pdf_pages, chunk_text


def test_chunk_text_uses_recursive_boundaries():
    """Prefer natural boundaries over splitting words mid-token."""
    text = (
        "LangChain splitters work best when they can break on spaces. "
        "This sentence should stay readable after chunking."
    )

    chunks = chunk_text(text, chunk_size=60, overlap=10)

    assert len(chunks) >= 2
    assert all(chunk.strip() == chunk for chunk in chunks)
    assert all(len(chunk) <= 60 for chunk in chunks)
    assert not any("splitters wo" in chunk and "rk best" in chunk for chunk in chunks)


def test_chunk_pdf_pages_preserves_metadata_per_chunk():
    """Chunked page records should keep source metadata and page-local indexes."""
    pages = [
        {
            "page_number": 3,
            "text": (
                "First paragraph on the page.\n\n"
                "Second paragraph adds enough content to force a split while "
                "keeping boundaries readable for retrieval."
            ),
        }
    ]

    chunks = chunk_pdf_pages(
        pages=pages,
        source_file="guide.pdf",
        source_path="data/uploads/guide.pdf",
        source_url="/uploads/guide.pdf",
        chunk_size=70,
        overlap=15,
    )

    assert len(chunks) >= 2
    assert [chunk["chunk_index"] for chunk in chunks] == list(range(len(chunks)))
    assert all(chunk["source_file"] == "guide.pdf" for chunk in chunks)
    assert all(chunk["source_path"] == "data/uploads/guide.pdf" for chunk in chunks)
    assert all(chunk["source_url"] == "/uploads/guide.pdf" for chunk in chunks)
    assert all(chunk["page_number"] == 3 for chunk in chunks)
