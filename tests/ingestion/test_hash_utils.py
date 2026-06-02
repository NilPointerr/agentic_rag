from app.ingestion.hash_utils import generate_content_hash, normalize_text


def test_content_hash_normalizes_text_before_hashing():
    """Hashes should be stable across casing and whitespace changes."""
    first = "  Hybrid\nSearch   Uses BM25  "
    second = "hybrid search uses bm25"

    assert normalize_text(first) == second
    assert generate_content_hash(first) == generate_content_hash(second)
