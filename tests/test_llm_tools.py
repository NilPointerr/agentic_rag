from app.llm_tools.llm_tools import vector_search_tool, web_search_tool


def test_vector_search_tool_returns_context_sources_and_score(monkeypatch):
    """Vector search tool should normalize retriever output for the agent."""
    monkeypatch.setattr(
        "app.llm_tools.llm_tools.retrieve",
        lambda query: ([{"text": "chunk one"}, {"text": "chunk two"}], 0.75),
    )

    result = vector_search_tool.invoke({"query": "what is rag"})

    assert result == {
        "context": ["chunk one", "chunk two"],
        "sources": [{"text": "chunk one"}, {"text": "chunk two"}],
        "score": 0.75,
    }


def test_web_search_tool_returns_search_results(monkeypatch):
    """Web search tool should delegate to the public search helper."""
    monkeypatch.setattr(
        "app.llm_tools.llm_tools.web_search",
        lambda query: [{"title": "Latest", "href": "https://example.com"}],
    )

    assert web_search_tool.invoke({"query": "latest rag news"}) == [
        {"title": "Latest", "href": "https://example.com"}
    ]
