import json
from app.retriever.retriever import retrieve
from app.llm.groq_client import generate_answer
from app.tools.web_search import web_search
from app.config.settings import settings
from app.utils.logger import logger

def rag_agent(query: str):
    threshold = settings.SIMILARITY_THRESHOLD

    messages = [
        {
            "role": "system",
            "content": f"""
    You are an AI agent with access to tools.

    You MUST follow this execution order:

    STEP 1:
    Always call vector_search first.

    After receiving vector_search results, you will also receive:
    - similarity_score (0 to 1)
    - retrieved_context

    STEP 2:
    Evaluate similarity_score.

    - If similarity_score >= {threshold}:
        Answer using retrieved_context.
        Do NOT call web_search.

    - If similarity_score < {threshold}:
        Call web_search to get better information.

    STEP 3:
    After web_search results, answer the user clearly.

    Rules:
    - Never answer from memory.
    - Always follow the step sequence.
    - Do not skip vector_search.
    """
        },
        {"role": "user", "content": query}
    ]

    while True:
        response = generate_answer(messages)
        message = response.choices[0].message

        # If no tool call, model is done.
        if not message.tool_calls:
            return message.content

        # Append assistant message that requested tool(s).
        messages.append(message)

        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name

            try:
                args = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON args for tool '{tool_name}', using empty args")
                args = {}

            tool_query = args.get("query", query)

            if tool_name == "vector_search":
                logger.info("Executing vector_search tool")
                context, score = retrieve(tool_query)
                retrieved_context = "\n".join(context) if context else "No relevant internal context found."
                tool_result = (
                    f"similarity_score: {score:.4f}\n"
                    f"threshold: {threshold:.4f}\n"
                    f"retrieved_context:\n{retrieved_context}"
                )
            elif tool_name == "web_search":
                logger.info("Executing web_search tool")
                tool_result = web_search(tool_query)
            else:
                logger.warning(f"Unknown tool called: {tool_name}")
                tool_result = f"Error: Unknown tool '{tool_name}'."

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result
            })
