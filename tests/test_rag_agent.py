from unittest.mock import patch

from app.agent.rag_agent import rag_agent


@patch("app.agent.rag_agent.generate_answer")
@patch("app.agent.rag_agent.retrieve")
def test_rag_agent_vector_only_path(mock_retrieve, mock_generate_answer):
    """Agent should avoid web search when retrieval score is above threshold."""

    mock_retrieve.return_value = {
        "passages": [
            {
                "text": "Context chunk",
                "score": 0.95,
                "source": "doc.pdf",
                "doc_id": "doc1",
                "chunk_id": 0,
                "page": 1,
            }
        ],
        "top_score": 0.95,
        "avg_score": 0.95,
        "used_top_k": 3,
        "used_min_score": 0.35,
    }

    class _Msg:
        content = "Answer from vector context"

    class _Choice:
        message = _Msg()

    class _Resp:
        choices = [_Choice()]

    mock_generate_answer.return_value = _Resp()

    result = rag_agent("Who is Shivaji?")

    assert result["decision"] == "vector_only"
    assert "Answer from vector context" in result["answer"]


@patch("app.agent.rag_agent.web_search")
@patch("app.agent.rag_agent.generate_answer")
@patch("app.agent.rag_agent.retrieve")
def test_rag_agent_web_fallback(mock_retrieve, mock_generate_answer, mock_web_search):
    """Agent should call web search when retrieval is below threshold."""

    mock_retrieve.return_value = {
        "passages": [],
        "top_score": 0.1,
        "avg_score": 0.1,
        "used_top_k": 3,
        "used_min_score": 0.35,
    }
    mock_web_search.return_value = [
        {"title": "Result", "snippet": "Snippet", "url": "https://example.com"}
    ]

    class _Msg:
        content = "Answer with web support"

    class _Choice:
        message = _Msg()

    class _Resp:
        choices = [_Choice()]

    mock_generate_answer.return_value = _Resp()

    result = rag_agent("What happened today?")

    assert result["decision"] == "vector_plus_web"
    assert len(result["citations"]) == 1
    assert result["citations"][0]["type"] == "web"
