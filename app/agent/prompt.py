def tool_decision_prompt(query: str):
    return f"""
You are an AI agent with access to tools.

TOOLS:
1. vector_search(query) → Search internal document database.
2. web_search(query) → Search the internet.
3. direct_answer() → Answer using your own knowledge (LAST RESORT).

IMPORTANT TOOL PRIORITY RULES:

- ALWAYS attempt vector_search FIRST.
- If the question could possibly be answered from stored documents, choose vector_search.
- Only choose web_search if the information is clearly not in stored documents OR is recent/current.
- Choose direct_answer ONLY if:
    - The question is extremely simple general knowledge
    - AND does not require documents
    - AND does not require up-to-date information

You should prefer tools over answering directly.

User Question:
{query}

Return ONLY a valid JSON object in this exact format:

{{
  "tool": "vector_search | web_search | direct_answer",
  "arguments": {{
      "query": "{query}"
  }}
}}

Rules:
- Do not explain.
- Do not add markdown.
- Do not add text before or after JSON.
- Always return valid JSON.
"""