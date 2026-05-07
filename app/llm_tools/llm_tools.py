from langchain.tools import tool
from app.retriever.retriever import retrieve
from app.search_tools.web_search import web_search

@tool
def vector_search_tool(query: str) -> dict:
    """
    Search the internal knowledge base using hybrid retrieval.

    Use this tool when the user question is likely answered by documents
    stored in the system's internal database (PDFs, manuals, reports,
    company knowledge base, etc.).

    This tool performs query expansion, BM25 lexical search, vector search,
    RRF merge, and reranking before returning the most relevant chunks.

    Args:
        query (str): The user question or search query.

    Returns:
        dict:
            context (list[str]): Retrieved document text chunks relevant
                                 to the query.
            score (float): Average similarity score (0–1) indicating how
                           relevant the retrieved context is.

    Guidance for the agent:
    - ALWAYS try this tool first for factual or document-based questions.
    """

    sources, score = retrieve(query)

    return {
        "context": [source["text"] for source in sources],
        "sources": sources,
        "score": score
    }


@tool
def web_search_tool(query: str) -> list[dict]:
    """
    Search the public internet for up-to-date information.

    Use this tool when:
    - The internal knowledge base does not contain relevant information.
    - The question requires recent or real-time data.
    - The retrieved internal context is not sufficient for the question.

    Examples of suitable queries:
    - Current news
    - Recent events
    - Latest technologies
    - Information not present in internal documents

    Args:
        query (str): The user search query.

    Returns:
        list[dict]: Structured web results containing title, snippet,
        and URL fields.
    """

    return web_search(query)
