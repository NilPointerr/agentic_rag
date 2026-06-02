from app.ingestion.chunker import chunk_pdf_pages, chunk_text, sentence_chunk


def test_chunk_text_uses_recursive_boundaries():
    """Prefer natural chunk boundaries and avoid splitting words mid-token."""
    text = (
        "LangChain splitters work best when they can break on spaces. "
        "This sentence should stay readable after chunking."
    )

    chunks = chunk_text(text, chunk_size=60, overlap=10)

    assert len(chunks) >= 2
    assert all(chunk.strip() == chunk for chunk in chunks)
    assert all(len(chunk) <= 60 for chunk in chunks)
    assert " ".join(chunks).replace(" .", ".") == text
    assert all(" " not in chunk[:1] for chunk in chunks[1:])


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


def test_chunk_pdf_pages_skips_blank_pages_and_empty_chunks():
    """Blank pages should not generate retrieval chunks."""
    pages = [
        {"page_number": 1, "text": "   "},
        {"page_number": 2, "text": "\n\n"},
        {"page_number": 3, "text": "usable text"},
    ]

    chunks = chunk_pdf_pages(
        pages=pages,
        source_file="guide.pdf",
        source_path="data/uploads/guide.pdf",
        source_url="/uploads/guide.pdf",
        chunk_size=100,
        overlap=10,
    )

    assert chunks == [
        {
            "text": "usable text",
            "source_file": "guide.pdf",
            "source_path": "data/uploads/guide.pdf",
            "source_url": "/uploads/guide.pdf",
            "page_number": 3,
            "chunk_index": 0,
        }
    ]


def test_sentence_chunk_respects_overlap(monkeypatch):
    """Sentence chunking should slide forward by max_sentences - overlap."""
    monkeypatch.setattr(
        "app.ingestion.chunker.nltk.sent_tokenize",
        lambda text: [
            "Sentence one.",
            "Sentence two.",
            "Sentence three.",
            "Sentence four.",
        ],
    )

    chunks = sentence_chunk("ignored", max_sentences=2, overlap=1)

    assert chunks == [
        "Sentence one. Sentence two.",
        "Sentence two. Sentence three.",
        "Sentence three. Sentence four.",
        "Sentence four.",
    ]
