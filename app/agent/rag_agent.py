from app.agent.langgraph_agent import graph


def rag_agent(query: str):
    """Invoke the compiled graph and normalize its answer payload."""
    result = graph.invoke({
        "query": query
    })

    return {
        "answer": result["answer"],
        "sources": result.get("sources", []),
    }

