from mcp.server.fastmcp import FastMCP 
from app.retriever.retriever import retrieve
from app.search_tools.web_search import web_search
from app.utils.logger import logger

mcp = FastMCP()

@mcp.tool()
async def vector_search(query: str) -> dict:
    logger.info(f"Received vector search query: {query}")
    """
    Search internal document database.
    """
    texts, score = retrieve(query)

    logger.info(f"Vector search returned {len(texts)} score: {score} results:- \n {texts}")

    return {
        "context": texts,
        "similarity_score": score
    }

@mcp.tool()
async def web_search_tool(query: str) -> str:
    """
    Search the internet for up-to-date information.
    """
    return web_search(query)