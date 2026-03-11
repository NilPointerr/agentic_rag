tools = [
    {
        "type": "function",
        "function": {
            "name": "vector_search",
            "description": "Search internal knowledge base documents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "User search query"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the internet for up-to-date information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "User search query"}
                },
                "required": ["query"],
            },
        },
    },
]
