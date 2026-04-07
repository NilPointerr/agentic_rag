# from typing import TypedDict, List
# from app.llm.groq_client import generate_answer
# from app.llm_tools.llm_tools import vector_search_tool, web_search_tool
# from langgraph.graph import StateGraph
# from app.utils.logger import logger


# class AgentState(TypedDict):
#     query: str
#     context: List[str]
#     score: float
#     answer: str


# def vector_node(state):
#     logger.info(f"Performing vector search for query: {state['query']}")


#     result = vector_search_tool.invoke({"query": state["query"]})

#     texts = result["context"]
#     score = result["score"]

#     logger.info(f"Vector search returned {len(texts)} results with average score {score}.") 
#     return {
#         "context": texts,
#         "score": score
#     }

# def decision_node(state):
#     if state["score"] > 0.65:
#         logger.info(f"Similarity score {state['score']} is above threshold. Proceeding to generate answer.")
#         return "generate_answer"
#     else:
#         logger.info(f"Similarity score {state['score']} is below threshold. Proceeding to web search.")
#         return "web_search"
    

# def web_node(state):
#     logger.info(f"Performing web search for query: {state['query']}")

#     results = web_search_tool.invoke({"query": state["query"]})

#     logger.info(f"Web search returned results.")
#     return {
#         "context": [results]
#     }


# def answer_node(state):

#     messages = [
#         {"role": "system", "content": "Answer using the provided context."},
#         {"role": "user", "content": state["query"]},
#         {"role": "assistant", "content": str(state["context"])}
#     ]

#     response = generate_answer(messages)

#     return {
#         "answer": response.choices[0].message.content
#     }


# builder = StateGraph(AgentState)

# builder.add_node("vector_search", vector_node)
# builder.add_node("web_search", web_node)
# builder.add_node("generate_answer", answer_node)

# builder.set_entry_point("vector_search")

# builder.add_conditional_edges(
#     "vector_search",
#     decision_node,
#     {
#         "generate_answer": "generate_answer",
#         "web_search": "web_search"
#     }
# )

# builder.add_edge("web_search", "generate_answer")

# graph = builder.compile()


from typing import Any, List, TypedDict
from langgraph.graph import StateGraph

from app.llm.groq_client import generate_answer
from app.llm_tools.llm_tools import vector_search_tool, web_search_tool
from app.utils.logger import logger


class AgentState(TypedDict):
    query: str
    context: List[str]
    sources: List[dict[str, Any]]
    use_web: bool
    answer: str


# -----------------------------
# Vector Search Node
# -----------------------------
def vector_node(state: AgentState):
    """Fetch internal vector-search context for the current query."""

    logger.info(f"Performing vector search for query: {state['query']}")

    result = vector_search_tool.invoke({
        "query": state["query"]
    })

    context = result["context"]
    sources = result.get("sources", [])

    logger.info(f"Vector search returned {len(context)} documents")

    return {
        "context": context,
        "sources": sources,
    }


# -----------------------------
# Context Evaluation Node
# -----------------------------
def evaluate_context_node(state: AgentState):
    """Ask the LLM whether retrieved internal context is sufficient."""

    logger.info("Evaluating context relevance")

    messages = [
        {
            "role": "system",
            "content": """
You are a context evaluator.

Determine whether the provided context is relevant enough to answer the user question.

Respond ONLY with:
YES
or
NO
"""
        },
        {
            "role": "system",
            "content": f"Context: {state['context']}"
        },
        {
            "role": "user",
            "content": state["query"]
        }
    ]

    response = generate_answer(messages)

    decision = response.choices[0].message.content.strip().upper()

    logger.info(f"Context evaluation result: {decision}")

    if decision == "YES":
        return {"use_web": False}
    else:
        return {"use_web": True}


# -----------------------------
# Web Search Node
# -----------------------------
def web_node(state: AgentState):
    """Fetch supplemental web results when internal context is insufficient."""

    logger.info("Performing web search")

    results = web_search_tool.invoke({
        "query": state["query"]
    })

    logger.info(f"Web search returned {len(results)} results")

    context = []

    for r in results[:5]:  # limit results for prompt size

        if isinstance(r, dict):
            title = r.get("title", "")
            body = r.get("body", "")
            href = r.get("href", "")
            text = (
                f"Title: {title}\n"
                f"Snippet: {body}\n"
                f"URL: {href}"
            )
        else:
            text = str(r)

        context.append(text)

    return {
        "context": context,
        "sources": [
            {
                "title": r.get("title", ""),
                "snippet": r.get("body", ""),
                "source_url": r.get("href", ""),
                "page_url": r.get("href", ""),
                "source_type": "web",
            }
            for r in results[:5]
            if isinstance(r, dict)
        ],
    }

# -----------------------------
# Answer Generation Node
# -----------------------------
def answer_node(state: AgentState):
    """Generate the final user-facing answer from the current context."""

    logger.info("Generating final answer")

    messages = [
        {
            "role": "system",
            "content": """
You are a helpful RAG assistant.

Use the provided context to answer the user clearly and completely.

Rules:
- Synthesize multiple context items into one coherent answer.
- Do not just copy a title or snippet.
- If the answer comes from web context, provide a concise overview first, then 2-5 key points.
- If URLs are present in the context, end with a short "Sources:" section listing them.
- If context is weak or incomplete, say so briefly.
"""
        },
        {
            "role": "system",
            "content": f"Context: {state['context']}"
        },
        {
            "role": "user",
            "content": state["query"]
        }
    ]

    response = generate_answer(messages)

    answer = response.choices[0].message.content

    logger.info("Answer generated successfully")

    return {
        "answer": answer
    }


# -----------------------------
# Routing Logic
# -----------------------------
def route_decision(state: AgentState):
    """Route execution to web search or answer generation."""

    if state["use_web"]:
        logger.info("Context not relevant -> switching to web search")
        return "web_search"

    logger.info("Context relevant -> generating answer")
    return "generate_answer"


# -----------------------------
# Build Graph
# -----------------------------
builder = StateGraph(AgentState)

builder.add_node("vector_search", vector_node)
builder.add_node("evaluate_context", evaluate_context_node)
builder.add_node("web_search", web_node)
builder.add_node("generate_answer", answer_node)

builder.set_entry_point("vector_search")

builder.add_edge("vector_search", "evaluate_context")

builder.add_conditional_edges(
    "evaluate_context",
    route_decision,
    {
        "web_search": "web_search",
        "generate_answer": "generate_answer"
    }
)

builder.add_edge("web_search", "generate_answer")

graph = builder.compile()
