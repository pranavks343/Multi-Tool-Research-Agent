"""Tool registry: schema→OpenAI conversion, the TOOLS list, and the dispatcher.

This module wires every tool together. It imports the individual tool functions and
their argument schemas, builds the OpenAI tool definitions, and exposes a single
`dispatch_tool()` entry point that validates args (via Pydantic) before running.
"""
from __future__ import annotations

import json
import sys

from pydantic import BaseModel, ValidationError

from app import cache
from app.config import DEBUG
from app.schemas import (
    CalculatorArgs,
    FetchYoutubeTranscriptArgs,
    SearchArxivArgs,
    SearchWebArgs,
    SearchWikipediaArgs,
)
from app.tools.arxiv import run_search_arxiv
from app.tools.calc import run_calculator
from app.tools.web import run_search_web
from app.tools.wiki import run_search_wikipedia
from app.tools.youtube import run_fetch_youtube_transcript


def pydantic_to_openai_tool(name: str, description: str, model: type[BaseModel]) -> dict:
    """Convert a Pydantic model into an OpenAI tool definition (strips title noise)."""
    schema = model.model_json_schema()
    schema.pop("title", None)
    for prop in schema.get("properties", {}).values():
        prop.pop("title", None)
    return {
        "type": "function",
        "function": {"name": name, "description": description, "parameters": schema},
    }

TOOLS = [
    pydantic_to_openai_tool(
        name="search_web",
        description=(
            "General-purpose web search. Use for news, blog posts, tutorials, opinions, "
            "current events, or anything non-academic and time-sensitive."
        ),
        model=SearchWebArgs,
    ),
    pydantic_to_openai_tool(
        name="search_arxiv",
        description=(
            "Search arXiv — an open-access repository of scholarly preprints. "
            "Returns titles, authors, abstracts, and arXiv IDs. Use for academic research, not general web."
        ),
        model=SearchArxivArgs,
    ),
    pydantic_to_openai_tool(
        name="search_wikipedia",
        description=(
            "Fetch a Wikipedia article summary. Best for stable encyclopedic facts: "
            "definitions, historical events, biographies, scientific concepts."
        ),
        model=SearchWikipediaArgs,
    ),
    pydantic_to_openai_tool(
        name="fetch_youtube_transcript",
        description=(
            "Retrieve the spoken-text transcript of a YouTube video. "
            "Use only when the user gives a specific video (URL or ID), not to discover videos on a topic."
        ),
        model=FetchYoutubeTranscriptArgs,
    ),
    pydantic_to_openai_tool(
        name="calculator",
        description=(
            "Safely evaluate a math expression. Supports arithmetic, exponentiation, "
            "parentheses, and standard math functions. Pure math evaluator — no code execution."
        ),
        model=CalculatorArgs,
    ),
]

TOOL_MODELS: dict[str, type[BaseModel]] = {
    "search_web": SearchWebArgs,
    "search_arxiv": SearchArxivArgs,
    "search_wikipedia": SearchWikipediaArgs,
    "fetch_youtube_transcript": FetchYoutubeTranscriptArgs,
    "calculator": CalculatorArgs,
}

TOOL_DISPATCH = {
    "search_web": run_search_web,
    "search_arxiv": run_search_arxiv,
    "search_wikipedia": run_search_wikipedia,
    "fetch_youtube_transcript": run_fetch_youtube_transcript,
    "calculator": run_calculator,
}


def _is_error(result: str) -> bool:
    """True if a tool result is a failure that must NOT be cached.

    Two failure shapes exist across the tools:
      - JSON tools return ``{"error": ..., "tool": ...}`` (see error_payload).
      - The calculator returns bare ``[error] ...`` / ``[calculator error] ...`` strings.

    The prefix check is deliberately narrow: the search tools return a JSON *array*
    (``[{...}]``) on success, which also starts with ``[`` — so a broad
    ``startswith("[")`` would wrongly treat every successful search as an error and
    never cache it.
    """
    if result.startswith(("[error]", "[calculator error]")):
        return True
    try:
        parsed = json.loads(result)
    except (json.JSONDecodeError, ValueError):
        return False
    return isinstance(parsed, dict) and "error" in parsed


def dispatch_tool(name: str, raw_args: dict) -> str:
    """Validate raw args against the Pydantic schema, check the cache, then run the tool."""
    if name not in TOOL_MODELS:
        return f"[error] Unknown tool: {name}"
    try:
        validated = TOOL_MODELS[name].model_validate(raw_args)
    except ValidationError as e:
        return f"[validation error] {e.errors()}"

    # Cache on the *validated* args: model_dump() normalizes defaults and types,
    # so cosmetically-different-but-equivalent calls share one entry.
    cache_args = validated.model_dump()

    # A cache hit skips the tool entirely.
    cached = cache.get(name, cache_args)
    if cached is not None:
        return cached

    result = TOOL_DISPATCH[name](validated)
    if result is None:
        result = json.dumps({"error": f"{name} returned None", "tool": name})

    # Store only successes — never poison the cache with a transient failure,
    # or every identical query would replay it for the full TTL.
    if not _is_error(result):
        cache.set(name, cache_args, result)

    if DEBUG:
        print(f"[TOOL DEBUG] {name}({raw_args}) -> {result[:300]}", file=sys.stderr)
    return result