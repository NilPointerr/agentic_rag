from langchain.tools import tool
from app.retriever.retriever import retrieve
from app.search_tools.web_search import web_search

@tool
def vector_search_tool(query: str) -> dict:
    """
    Search the internal knowledge base using vector similarity.

    Use this tool when the user question is likely answered by documents
    stored in the system's internal database (PDFs, manuals, reports,
    company knowledge base, etc.).

    This tool performs semantic search on the vector database and returns
    the most relevant document chunks along with a similarity score.

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
    - If the similarity score is low (< 0.75), consider using web_search_tool
      to fetch more up-to-date information.
    """

    texts, score = retrieve(query)

    return {
        "context": texts,
        "score": score
    }


@tool
def web_search_tool(query: str) -> str:
    """
    Search the public internet for up-to-date information.

    Use this tool when:
    - The internal knowledge base does not contain relevant information.
    - The question requires recent or real-time data.
    - The vector_search_tool returns a low similarity score.

    Examples of suitable queries:
    - Current news
    - Recent events
    - Latest technologies
    - Information not present in internal documents

    Args:
        query (str): The user search query.

    Returns:
        str: A summarized collection of web search results including
             titles, snippets, and URLs.
    """

    return web_search(query)