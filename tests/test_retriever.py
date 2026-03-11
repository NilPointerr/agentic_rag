from unittest.mock import Mock, patch

from app.retriever.retriever import retrieve


@patch("app.retriever.retriever.embed_texts")
@patch("app.retriever.retriever.get_index")
def test_retrieve_uses_top_k_and_filters_by_score(mock_get_index, mock_embed_texts):
    """Retriever should honor top_k and filter low-score matches."""

    mock_embed_texts.return_value = [[0.1, 0.2, 0.3]]

    mock_index = Mock()
    mock_index.query.return_value = {
        "matches": [
            {
                "score": 0.82,
                "metadata": {
                    "text": "High confidence chunk",
                    "source": "doc-a.pdf",
                    "doc_id": "doc-a",
                    "chunk_id": 0,
                    "page": 1,
                },
            },
            {
                "score": 0.2,
                "metadata": {
                    "text": "Low confidence chunk",
                    "source": "doc-a.pdf",
                    "doc_id": "doc-a",
                    "chunk_id": 1,
                    "page": 1,
                },
            },
        ]
    }
    mock_get_index.return_value = mock_index

    result = retrieve("test query", top_k=5, min_score=0.3)

    assert result["used_top_k"] == 5
    assert len(result["passages"]) == 1
    assert result["top_score"] == 0.82
    assert result["passages"][0]["text"] == "High confidence chunk"
