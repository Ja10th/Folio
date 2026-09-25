"""The three tools, declared the way a model API expects them.

Gemini's function-calling docs use this shape: a name, a description,
and a JSON schema for the arguments. The graph calls the same functions.
A model would return functionCall parts; Folio records the same objects.
"""

from __future__ import annotations

DECLARATIONS = [
    {
        "name": "search_web",
        "description": "Search the public web. Returns titles, URLs, and snippets.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "How many hits to return.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "fetch_page",
        "description": "Download a public page and extract its readable text.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Absolute http or https URL of the page to read.",
                },
            },
            "required": ["url"],
        },
    },
    {
        "name": "summarize",
        "description": "Condense fetched pages into sourced claims. Does not add facts that are not in the pages.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The question the brief must answer.",
                },
                "documents": {
                    "type": "integer",
                    "description": "How many fetched documents to condense.",
                },
            },
            "required": ["query", "documents"],
        },
    },
]


def function_call(name: str, args: dict, call_id: str) -> dict:
    """A functionCall part, in the shape Gemini returns to the app."""
    return {"id": call_id, "name": name, "args": args}


def function_response(name: str, call_id: str, response: dict) -> dict:
    """A functionResponse part, which the app sends back after running the tool."""
    return {"id": call_id, "name": name, "response": response}
