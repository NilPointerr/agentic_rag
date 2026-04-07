from io import BytesIO

import pytest
from fastapi import HTTPException
from starlette.datastructures import UploadFile
from starlette.requests import Request

from app.api.routes import QueryRequest, ingest_documents, query_agent
from app.config.settings import settings
from app.retriever import retriever
from app.search_tools import web_search


@pytest.fixture
def fake_request():
    return Request({"type": "http", "method": "POST", "path": "/", "headers": []})


def test_retrieve_filters_matches_by_threshold(monkeypatch):
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

    texts, score = retriever.retrieve("test query", top_k=2)

    assert texts == ["high confidence"]
    assert score == 0.92


def test_web_search_returns_structured_results(monkeypatch):
    class FakeDDGS:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def text(self, query, max_results):
            return [
                {"title": "Result 1", "body": "Snippet 1", "href": "https://example.com/1"},
                {"title": "Result 2", "body": "Snippet 2", "href": "https://example.com/2"},
            ]

    monkeypatch.setattr(web_search, "DDGS", FakeDDGS, raising=False)

    results = web_search.web_search("agentic rag")

    assert results == [
        {"title": "Result 1", "body": "Snippet 1", "href": "https://example.com/1"},
        {"title": "Result 2", "body": "Snippet 2", "href": "https://example.com/2"},
    ]


def test_query_endpoint_rejects_long_queries(fake_request):
    long_query = "x" * (settings.MAX_QUERY_LENGTH + 1)

    with pytest.raises(HTTPException) as exc_info:
        query_agent(fake_request, QueryRequest(query=long_query))

    assert exc_info.value.status_code == 400
    assert "character limit" in exc_info.value.detail


@pytest.mark.anyio
async def test_ingest_endpoint_rejects_path_traversal_filename(monkeypatch, fake_request):
    captured = {}

    monkeypatch.setattr("app.api.routes.load_pdf", lambda path: "safe text")
    monkeypatch.setattr("app.api.routes.chunk_text", lambda text: ["chunk"])
    monkeypatch.setattr("app.api.routes.embed_and_store", lambda chunks: chunks)

    def fake_open(path, mode):
        captured["path"] = str(path)

        class DummyFile:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def write(self, chunk):
                return len(chunk)

        return DummyFile()

    monkeypatch.setattr("builtins.open", fake_open)

    upload = UploadFile(filename="../../etc/passwd.pdf", file=BytesIO(b"%PDF-1.4 test"))
    response = await ingest_documents(fake_request, upload)

    assert response["chunks_created"] == 1
    assert ".." not in captured["path"]
    assert captured["path"].startswith("data/uploads/")


def test_auth_dependency_enforces_bearer_token_when_enabled():
    from app.security import verify_bearer_token

    original_auth_enabled = settings.AUTH_ENABLED
    original_secret = settings.JWT_SECRET_KEY

    settings.AUTH_ENABLED = True
    settings.JWT_SECRET_KEY = "secret"

    try:
        with pytest.raises(HTTPException):
            verify_bearer_token(None)
    finally:
        settings.AUTH_ENABLED = original_auth_enabled
        settings.JWT_SECRET_KEY = original_secret
