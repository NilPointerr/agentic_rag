from app.agent.langgraph_agent import graph

def rag_agent(query: str):

    result = graph.invoke({
        "query": query
    })

    return result["answer"]

