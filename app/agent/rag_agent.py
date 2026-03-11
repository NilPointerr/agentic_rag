from __future__ import annotations

from typing import Any

from app.config.settings import settings
from app.llm.groq_client import generate_answer
from app.retriever.retriever import retrieve
from app.tools.web_search import web_search
from app.utils.logger import logger


def _build_context(passages: list[dict[str, Any]]) -> str:
    """Build LLM context text from retrieved passages."""

    blocks: list[str] = []
    for idx, passage in enumerate(passages, start=1):
        blocks.append(
            f"[{idx}] source={passage.get('source')} page={passage.get('page')} "
            f"score={passage.get('score', 0):.3f}\n{passage.get('text', '')}"
        )
    return "\n\n".join(blocks)


def _build_web_context(results: list[dict[str, str]]) -> str:
    """Build LLM context text from web search results."""

    blocks: list[str] = []
    for idx, result in enumerate(results, start=1):
        blocks.append(
            f"[{idx}] title={result.get('title', '')}\n"
            f"snippet={result.get('snippet', '')}\n"
            f"url={result.get('url', '')}"
        )
    return "\n\n".join(blocks)


def _format_citations(passages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert retrieval passages into API citation payloads."""

    return [
        {
            "type": "vector",
            "source": item.get("source"),
            "doc_id": item.get("doc_id"),
            "chunk_id": item.get("chunk_id"),
            "page": item.get("page"),
            "score": item.get("score"),
        }
        for item in passages
    ]


def _format_web_citations(results: list[dict[str, str]]) -> list[dict[str, str]]:
    """Convert web results into API citation payloads."""

    return [
        {
            "type": "web",
            "source": row.get("title", ""),
            "url": row.get("url", ""),
        }
        for row in results
    ]


def rag_agent(query: str) -> dict[str, Any]:
    """Answer query using deterministic retrieval-first orchestration."""

    max_steps = settings.MAX_AGENT_STEPS
    step = 0

    step += 1
    retrieval = retrieve(query=query)
    passages = retrieval["passages"]
    top_score = float(retrieval["top_score"])
    threshold = settings.SIMILARITY_THRESHOLD

    citations = _format_citations(passages)
    context = _build_context(passages)
    decision = "vector_only"

    if step >= max_steps:
        return {
            "answer": "I could not complete the request within execution limits.",
            "citations": citations,
            "decision": "max_steps_reached",
            "metrics": {"top_score": top_score, "threshold": threshold, "steps": step},
        }

    web_results: list[dict[str, str]] = []
    if top_score < threshold:
        step += 1
        decision = "vector_plus_web"
        logger.info("Retrieval below threshold %.3f < %.3f", top_score, threshold)
        web_results = web_search(query=query)
        citations.extend(_format_web_citations(web_results))

    if step >= max_steps:
        return {
            "answer": "I gathered evidence but could not finish synthesis in time.",
            "citations": citations,
            "decision": "max_steps_reached",
            "metrics": {"top_score": top_score, "threshold": threshold, "steps": step},
        }

    step += 1
    prompt_context = context
    if web_results:
        prompt_context = (
            f"Internal context:\n{context or 'None'}\n\n"
            f"Web context:\n{_build_web_context(web_results)}"
        )

    messages = [
        {
            "role": "system",
            "content": (
                "Answer only from provided context. "
                "If context is insufficient, say uncertainty clearly. "
                "Prefer concise and factual responses."
            ),
        },
        {
            "role": "user",
            "content": f"Question: {query}\n\nContext:\n{prompt_context}",
        },
    ]

    response = generate_answer(messages)
    answer = response.choices[0].message.content if response.choices else ""

    return {
        "answer": answer or "No answer generated.",
        "citations": citations,
        "decision": decision,
        "metrics": {"top_score": top_score, "threshold": threshold, "steps": step},
    }
