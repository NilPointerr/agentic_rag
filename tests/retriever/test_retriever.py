from app.retriever import bm25_store
from app.retriever import retriever


def test_retrieve_returns_all_matches(monkeypatch):
    """Ensure retrieval preserves all Pinecone matches without threshold filtering."""
    monkeypatch.setattr(
        retriever,
        "embed_texts",
        lambda texts: [[0.1, 0.2, 0.3]],
    )

    class FakeIndex:
        def query(self, **kwargs):
            return {
                "matches": [
                    {"score": 0.92, "metadata": {"text": "high confidence"}},
                    {"score": 0.40, "metadata": {"text": "low confidence"}},
                ]
            }

    monkeypatch.setattr(retriever, "get_index", lambda: FakeIndex())
    monkeypatch.setattr(retriever.settings, "RERANK_ENABLED", False)
    monkeypatch.setattr(retriever.settings, "HYBRID_SEARCH_ENABLED", False)

    sources, score = retriever.retrieve("test query", top_k=2)

    assert sources == [
        {
            "text": "high confidence",
            "score": 0.92,
            "source_file": None,
            "source_path": None,
            "source_url": None,
            "page_number": None,
            "page_url": None,
            "chunk_index": None,
        },
        {
            "text": "low confidence",
            "score": 0.40,
            "source_file": None,
            "source_path": None,
            "source_url": None,
            "page_number": None,
            "page_url": None,
            "chunk_index": None,
        },
    ]
    assert score == 0.66


def test_retrieve_reranks_results(monkeypatch):
    """Ensure reranking can reorder Pinecone results before returning context."""
    monkeypatch.setattr(
        retriever,
        "embed_texts",
        lambda texts: [[0.1, 0.2, 0.3]],
    )

    class FakeIndex:
        def query(self, **kwargs):
            return {
                "matches": [
                    {"score": 0.95, "metadata": {"text": "vector winner"}},
                    {"score": 0.70, "metadata": {"text": "rerank winner"}},
                ]
            }

    class FakeReranker:
        def predict(self, pairs):
            return [0.2, 0.9]

    monkeypatch.setattr(retriever, "get_index", lambda: FakeIndex())
    monkeypatch.setattr(retriever.settings, "RERANK_ENABLED", True)
    monkeypatch.setattr(retriever.settings, "RERANK_CANDIDATES", 8)
    monkeypatch.setattr(retriever.settings, "HYBRID_SEARCH_ENABLED", False)
    monkeypatch.setattr(retriever, "get_reranker", lambda: FakeReranker())

    sources, score = retriever.retrieve("test query", top_k=2)

    assert [source["text"] for source in sources] == ["rerank winner", "vector winner"]
    assert sources[0]["vector_score"] == 0.70
    assert sources[0]["rerank_score"] == 0.9
    assert score == 0.55


def test_retrieve_falls_back_when_reranker_unavailable(monkeypatch):
    """Ensure retrieval still succeeds if the reranker cannot be loaded."""
    monkeypatch.setattr(
        retriever,
        "embed_texts",
        lambda texts: [[0.1, 0.2, 0.3]],
    )

    class FakeIndex:
        def query(self, **kwargs):
            return {
                "matches": [
                    {"score": 0.8, "metadata": {"text": "first"}},
                    {"score": 0.6, "metadata": {"text": "second"}},
                ]
            }

    monkeypatch.setattr(retriever, "get_index", lambda: FakeIndex())
    monkeypatch.setattr(retriever.settings, "RERANK_ENABLED", True)
    monkeypatch.setattr(retriever.settings, "HYBRID_SEARCH_ENABLED", False)
    monkeypatch.setattr(
        retriever,
        "get_reranker",
        lambda: (_ for _ in ()).throw(RuntimeError("missing model")),
    )

    sources, score = retriever.retrieve("test query", top_k=2)

    assert [source["text"] for source in sources] == ["first", "second"]
    assert score == 0.7


def test_retrieve_merges_vector_and_bm25_with_rrf(monkeypatch, tmp_path):
    """Ensure hybrid retrieval deduplicates vector and BM25 candidates."""
    monkeypatch.setattr(
        retriever,
        "embed_texts",
        lambda texts: [[0.1, 0.2, 0.3]],
    )
    monkeypatch.setattr(
        bm25_store.settings,
        "BM25_INDEX_PATH",
        str(tmp_path / "bm25_chunks.jsonl"),
    )
    monkeypatch.setattr(retriever.settings, "RERANK_ENABLED", False)
    monkeypatch.setattr(retriever.settings, "HYBRID_SEARCH_ENABLED", True)
    monkeypatch.setattr(retriever.settings, "RETRIEVAL_CANDIDATES", 20)

    shared_chunk = {
        "text": "hybrid retrieval combines bm25 and vector search",
        "source_file": "hybrid.pdf",
        "page_number": 1,
        "chunk_index": 0,
    }
    shared_chunk["chunk_id"] = bm25_store.build_chunk_id(shared_chunk)
    bm25_store.upsert_documents([shared_chunk])

    class FakeIndex:
        def query(self, **kwargs):
            assert kwargs["top_k"] == 20
            return {
                "matches": [
                    {
                        "score": 0.88,
                        "metadata": shared_chunk,
                    },
                ]
            }

    monkeypatch.setattr(retriever, "get_index", lambda: FakeIndex())

    sources, score = retriever.retrieve("bm25 vector hybrid", top_k=5)

    assert len(sources) == 1
    assert sources[0]["chunk_id"] == shared_chunk["chunk_id"]
    assert sources[0]["vector_score"] == 0.88
    assert sources[0]["bm25_score"] > 0
    assert sources[0]["rrf_score"] == score


def test_build_source_from_metadata_sets_page_url():
    """Page URLs should include the page anchor when both values exist."""
    source = retriever._build_source_from_metadata(
        {
            "text": "chunk",
            "source_url": "/uploads/example.pdf",
            "page_number": 2,
            "chunk_id": "chunk-1",
        },
        score=0.8,
    )

    assert source["page_url"] == "/uploads/example.pdf#page=2"
    assert source["chunk_id"] == "chunk-1"


def test_expand_query_returns_original_when_disabled(monkeypatch):
    """Query expansion should short-circuit when the feature is disabled."""
    monkeypatch.setattr(retriever.settings, "QUERY_EXPANSION_ENABLED", False)
    assert retriever.expand_query("hybrid search") == "hybrid search"


def test_rrf_merge_handles_single_source_lists():
    """RRF merge should gracefully handle vector-only and BM25-only input."""
    vector_only = retriever.rrf_merge([{"text": "v", "score": 0.8}], [], top_k=1)
    bm25_only = retriever.rrf_merge([], [{"text": "b", "score": 0.6}], top_k=1)

    assert vector_only == [{"text": "v", "score": 0.8}]
    assert bm25_only == [{"text": "b", "score": 0.6}]
