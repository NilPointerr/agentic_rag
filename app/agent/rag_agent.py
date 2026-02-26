import json
from app.retriever.retriever import retrieve
from app.llm.groq_client import generate_answer
from app.tools.web_search import web_search
from app.agent.prompt import tool_decision_prompt
from app.utils.logger import logger

def rag_agent(query: str):

    messages = [
            {
            "role": "system",
            "content": """
        You are an AI agent with tool access.

        Rules:
        - For ANY factual question about people, history, geography, politics, or numbers → ALWAYS use a tool.
        - Use vector_search for historical or document-based info.
        - Use web_search for current or recent info.
        - Do NOT answer from memory if a tool is available.
        """
        },
        {"role": "user", "content": query}
    ]

    #     messages = [
    #     {
    #         "role": "system",
    #         "content": """
    # You are an AI agent with access to tools.

    # You MUST follow this execution order:

    # STEP 1:
    # Always call vector_search first.

    # After receiving vector_search results, you will also receive:
    # - similarity_score (0 to 1)
    # - retrieved_context

    # STEP 2:
    # Evaluate similarity_score.

    # - If similarity_score >= 0.75:
    #     Answer using retrieved_context.
    #     Do NOT call web_search.

    # - If similarity_score < 0.75:
    #     Call web_search to get better information.

    # STEP 3:
    # After web_search results, answer the user clearly.

    # Rules:
    # - Never answer from memory.
    # - Always follow the step sequence.
    # - Do not skip vector_search.
    # """
    #     },
    #     {"role": "user", "content": query}
    # ]


    response = generate_answer(messages)
    message = response.choices[0].message

    # 🔥 If model wants to call tool
    if message.tool_calls:
        tool_call = message.tool_calls[0]
        tool_name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)

        if tool_name == "vector_search":
            logger.info("Executing vector_search tool")
            context, score = retrieve(args["query"])
            tool_result = "\n".join(context)

        elif tool_name == "web_search":
            logger.info("Executing web_search tool")
            tool_result = web_search(args["query"])

        # Send tool result back
        messages.append(message)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": tool_result
        })

        # Final answer
        final_response = generate_answer(messages)
        return final_response.choices[0].message.content

    # If no tool used
    return message.content