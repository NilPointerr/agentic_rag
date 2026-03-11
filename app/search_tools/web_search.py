from ddgs import DDGS
from app.utils.logger import logger


def web_search(query: str, max_results=5):
    logger.info(f"Performing web search for query: {query}")

    results_text = []

    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))

    logger.info(f"Web search returned {len(results)} results:- \n {results}")

    for r in results:
        results_text.append(
            f"Title: {r.get('title', '')}\n"
            f"Snippet: {r.get('body', '')}\n"
            f"URL: {r.get('href', '')}\n"
        )

    return "\n".join(results_text)