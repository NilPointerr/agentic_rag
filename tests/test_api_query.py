import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.api.routes import QueryRequest, query_agent


def test_query_endpoint_returns_citations_and_metrics():
    """Query handler should return answer, citations, decision, and metrics."""

    request = QueryRequest(query="test")
    fake_http_request = SimpleNamespace(client=SimpleNamespace(host="127.0.0.1"))

    fake_response = {
        "answer": "Test answer",
        "citations": [{"type": "vector", "source": "doc.pdf"}],
        "decision": "vector_only",
        "metrics": {"top_score": 0.9, "threshold": 0.65, "steps": 2},
    }

    with patch(
        "app.api.routes.run_in_threadpool", new=AsyncMock(return_value=fake_response)
    ):
        result = asyncio.run(query_agent(request, fake_http_request))

    assert result.answer == "Test answer"
    assert result.decision == "vector_only"
    assert isinstance(result.citations, list)
    assert "top_score" in result.metrics
