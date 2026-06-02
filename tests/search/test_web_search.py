from app.search_tools import web_search


def test_web_search_returns_structured_results(monkeypatch):
    """Ensure web search output is normalized into title/body/href dictionaries."""

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

    assert web_search.web_search("agentic rag") == [
        {"title": "Result 1", "body": "Snippet 1", "href": "https://example.com/1"},
        {"title": "Result 2", "body": "Snippet 2", "href": "https://example.com/2"},
    ]


def test_web_image_search_returns_structured_results(monkeypatch):
    """Ensure image search output is normalized into frontend-friendly fields."""

    class FakeDDGS:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def images(self, query, max_results):
            return [
                {
                    "title": "Result image",
                    "image": "https://img.example.com/full.jpg",
                    "thumbnail": "https://img.example.com/thumb.jpg",
                    "url": "https://example.com/page",
                    "source": "Example",
                }
            ]

    monkeypatch.setattr(web_search, "DDGS", FakeDDGS, raising=False)

    assert web_search.web_image_search("agentic rag") == [
        {
            "title": "Result image",
            "image_url": "https://img.example.com/full.jpg",
            "thumbnail_url": "https://img.example.com/thumb.jpg",
            "source_url": "https://example.com/page",
            "source": "Example",
        }
    ]
