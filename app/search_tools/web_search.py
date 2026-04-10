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


def web_image_search(query: str, max_results=6):
    """Run an image search and normalize results for frontend rendering."""
    logger.info(f"Performing image search for query: {query}")

    try:
        with DDGS() as ddgs:
            results = list(ddgs.images(query, max_results=max_results))
    except Exception as exc:
        logger.warning(f"Image search failed for query '{query}': {exc}")
        return []

    logger.info(f"Image search returned {len(results)} results")

    structured_results = []
    for r in results:
        image_url = r.get("image") or r.get("thumbnail")
        thumbnail_url = r.get("thumbnail") or image_url

        if not image_url:
            continue

        structured_results.append(
            {
                "title": r.get("title", ""),
                "image_url": image_url,
                "thumbnail_url": thumbnail_url,
                "source_url": r.get("url", "") or r.get("source", ""),
                "source": r.get("source", ""),
            }
        )

    return structured_results
