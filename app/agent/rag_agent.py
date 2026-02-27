import os
import json
import ast
import re

from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client

from app.llm.groq_client import generate_answer
from app.retriever.retriever import retrieve
from app.search_tools.web_search import web_search
from app.utils.logger import logger

def _extract_tool_text(tool_result) -> str:
    parts = []
    for block in getattr(tool_result, "content", []) or []:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "\n".join(parts).strip()


def _extract_vector_payload(tool_result) -> tuple[list[str], float]:
    # 1) Preferred path for MCP structured payloads.
    payload = getattr(tool_result, "structuredContent", None)
    if payload is None:
        payload = getattr(tool_result, "structured_content", None)

    # 2) Fallback for text-only tool payloads (JSON or Python-dict string).
    if payload is None:
        text_payload = _extract_tool_text(tool_result)
        if text_payload:
            try:
                payload = json.loads(text_payload)
            except Exception:
                try:
                    payload = ast.literal_eval(text_payload)
                except Exception:
                    payload = None

    if not isinstance(payload, dict):
        return [], 0.0

    context = payload.get("context", [])
    score = payload.get("similarity_score", 0.0)

    if isinstance(context, str):
        context = [context]
    if not isinstance(context, list):
        context = []

    try:
        score = float(score)
    except Exception:
        score = 0.0

    return context, score


def _parse_json_object(text: str) -> dict | None:
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def _should_use_web_search(query: str, vector_context: list[str], similarity_score: float) -> bool:
    context_preview = "\n".join(vector_context[:3]) if vector_context else "N/A"
    decision_prompt = [
        {
            "role": "system",
            "content": (
                "You are a retrieval planner. Decide whether web search is required.\n"
                "Return ONLY JSON: {\"use_web_search\": true|false, \"reason\": \"short\"}.\n"
                "Choose true when user asks for recent/current info or vector context is missing/insufficient."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {query}\n"
                f"Similarity score: {similarity_score}\n"
                f"Vector context preview:\n{context_preview}"
            ),
        },
    ]
    try:
        response = generate_answer(decision_prompt, use_tools=False)
        raw = response.choices[0].message.content or ""
        parsed = _parse_json_object(raw) or {}
        use_web = bool(parsed.get("use_web_search", False))
        logger.info(f"Web-search decision: {use_web} | raw: {raw}")
        return use_web
    except Exception as exc:
        logger.warning("Web-search decision failed, using fallback: %s", exc)
        return len(vector_context) == 0


def _answer_from_context(query: str, vector_context: list[str], web_context: str = "") -> str:
    prompt = [
        {
            "role": "system",
            "content": (
                "You are a concise assistant. Use provided context only.\n"
                "Return markdown with these exact sections:\n"
                "## Direct Answer\n"
                "## Key Points\n"
                "## Evidence Used\n"
                "If context is insufficient, explicitly say so in Direct Answer."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {query}\n\n"
                f"Vector Context:\n{chr(10).join(vector_context) if vector_context else 'N/A'}\n\n"
                f"Web Context:\n{web_context if web_context else 'N/A'}"
            ),
        },
    ]
    response = generate_answer(prompt, use_tools=False)
    return response.choices[0].message.content


async def _rag_with_local_tools(query: str) -> str:
    vector_context, similarity_score = retrieve(query)
    web_context = ""
    if _should_use_web_search(query, vector_context, float(similarity_score)):
        web_context = web_search(query)
    return _answer_from_context(query, vector_context, web_context)


async def _rag_with_mcp(query: str, mcp_server_url: str) -> str:
    async with streamable_http_client(mcp_server_url) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            vector_result = await session.call_tool("vector_search", {"query": query})
            vector_context, similarity_score = _extract_vector_payload(vector_result)

            # Last-resort safety: if MCP payload shape is unexpected, still use vector DB directly.
            if not vector_context and similarity_score == 0.0:
                logger.warning("MCP vector_search payload empty/unparsed. Falling back to direct retrieve().")
                vector_context, similarity_score = retrieve(query)

            logger.info(f"vector_context: {vector_context}")
            logger.info(f"similarity_score: {similarity_score}")

            web_context = ""
            if _should_use_web_search(query, vector_context, float(similarity_score)):
                logger.info("Performing web search...")
                web_result = await session.call_tool("web_search_tool", {"query": query})
                web_context = _extract_tool_text(web_result)

            logger.info(f"web_context: {web_context}")
            return _answer_from_context(query, vector_context, web_context)


async def rag_agent(query: str) -> str:
    mcp_server_url = os.getenv("MCP_SERVER_URL", "").strip()
    logger.info(f"MCP_SERVER_URL: {mcp_server_url}")
    if not mcp_server_url:
        logger.info("MCP_SERVER_URL not set; using local tools directly.")
        return await _rag_with_local_tools(query)

    try:
        logger.info("Attempting to connect to MCP server...")
        result = await _rag_with_mcp(query, mcp_server_url)
        logger.info("Successfully got response from MCP server.")
        return result
    except Exception as exc:
        logger.warning("MCP path failed, falling back to local tools: %s", exc)
        return await _rag_with_local_tools(query)
