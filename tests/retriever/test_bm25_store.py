import json

import pytest

from app.retriever import bm25_store
from app.retriever.bm25_store import BM25IndexError, load_documents, search_bm25, tokenize


def test_tokenize_normalizes_case_and_punctuation():
    """Tokenization should keep lowercase alphanumeric terms only."""
    assert tokenize("BM25, Hybrid-search! 123") == ["bm25", "hybrid", "search", "123"]


def test_bm25_store_upserts_and_searches_chunks(monkeypatch, tmp_path):
    """Ensure the local BM25 corpus returns lexical matches with metadata."""
    monkeypatch.setattr(
        bm25_store.settings,
        "BM25_INDEX_PATH",
        str(tmp_path / "bm25_chunks.jsonl"),
    )

    chunks = [
        {
            "text": "pinecone vector search uses embeddings",
            "source_file": "vector.pdf",
            "page_number": 1,
            "chunk_index": 0,
        },
        {
            "text": "okapi bm25 lexical search uses token frequency",
            "source_file": "bm25.pdf",
            "page_number": 2,
            "chunk_index": 0,
        },
    ]

    assert bm25_store.upsert_documents(chunks) == 2

    results = bm25_store.search_bm25("lexical token frequency", top_k=1)

    assert len(results) == 1
    assert results[0].document.metadata["source_file"] == "bm25.pdf"
    assert results[0].score > 0


def test_load_documents_skips_invalid_rows(monkeypatch, tmp_path):
    """Invalid JSONL rows should be ignored instead of breaking the corpus."""
    path = tmp_path / "bm25_chunks.jsonl"
    path.write_text(
        json.dumps({"text": "valid row", "source_file": "guide.pdf"}) + "\n"
        + "{not-json}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(bm25_store.settings, "BM25_INDEX_PATH", str(path))

    documents = load_documents()

    assert len(documents) == 1
    assert documents[0].text == "valid row"


def test_search_bm25_returns_empty_for_blank_query(monkeypatch, tmp_path):
    """Blank or punctuation-only queries should return no BM25 results."""
    monkeypatch.setattr(
        bm25_store.settings,
        "BM25_INDEX_PATH",
        str(tmp_path / "bm25_chunks.jsonl"),
    )

    assert search_bm25("!!!", top_k=5) == []
    assert search_bm25("valid", top_k=0) == []


def test_upsert_documents_raises_on_directory_creation_failure(monkeypatch, tmp_path):
    """Filesystem failures should surface as BM25 index errors."""
    monkeypatch.setattr(
        bm25_store.settings,
        "BM25_INDEX_PATH",
        str(tmp_path / "nested" / "bm25_chunks.jsonl"),
    )
    monkeypatch.setattr(
        "pathlib.Path.mkdir",
        lambda self, parents, exist_ok: (_ for _ in ()).throw(OSError("denied")),
    )

    with pytest.raises(BM25IndexError):
        bm25_store.upsert_documents([{"text": "chunk"}])
