from ddgs import DDGS
from app.utils.logger import logger


def web_search(query: str, max_results=5):
    """Run a web search and normalize results into a structured shape."""
    logger.info(f"Performing web search for query: {query}")

    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))

    logger.info(f"Web search returned {len(results)} results:- \n {results}")

    structured_results = []
    for r in results:
        structured_results.append(
            {
                "title": r.get("title", ""),
                "body": r.get("body", ""),
                "href": r.get("href", ""),
            }
        )

    return structured_results
