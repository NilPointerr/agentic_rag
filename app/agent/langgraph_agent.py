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


from typing import TypedDict, List
from langgraph.graph import StateGraph

from app.llm.groq_client import generate_answer
from app.llm_tools.llm_tools import vector_search_tool, web_search_tool
from app.utils.logger import logger


class AgentState(TypedDict):
    query: str
    context: List[str]
    use_web: bool
    answer: str


# -----------------------------
# Vector Search Node
# -----------------------------
def vector_node(state: AgentState):

    logger.info(f"Performing vector search for query: {state['query']}")

    result = vector_search_tool.invoke({
        "query": state["query"]
    })

    context = result["context"]

    logger.info(f"Vector search returned {len(context)} documents")

    return {
        "context": context
    }


# -----------------------------
# Context Evaluation Node
# -----------------------------
def evaluate_context_node(state: AgentState):

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

    logger.info("Performing web search")

    results = web_search_tool.invoke({
        "query": state["query"]
    })

    logger.info(f"Web search returned {len(results)} results")

    context = []

    for r in results[:5]:  # limit results for prompt size

        # Case 1: result is dictionary
        if isinstance(r, dict):
            title = r.get("title", "")
            body = r.get("body", "")
            text = f"{title}: {body}"

        # Case 2: result is string
        else:
            text = str(r)

        context.append(text)

    return {
        "context": context
    }

# -----------------------------
# Answer Generation Node
# -----------------------------
def answer_node(state: AgentState):

    logger.info("Generating final answer")

    messages = [
        {
            "role": "system",
            "content": "Answer the user using the provided context."
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