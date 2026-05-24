from __future__ import annotations
import json
import os
import sys
from datetime import datetime
from typing import Literal, Optional
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

from ddgs import DDGS
load_dotenv(find_dotenv())


class SearchWebArgs(BaseModel):
    """Arguments for a general web search."""

    query: str = Field(
        ...,
        description="The search query. Free-form natural language.",
    )
    max_results: int = Field(
        default=5, ge=1, le=20,
        description="Number of results to return.",
    )
    recency: Optional[Literal["day", "week", "month", "year"]] = Field(
        default=None,
        description=(
            "Restrict results to the recent past. Omit for no time filter. "
            "Use when the user asks for 'recent' or 'latest' information."
        ),
    )


ArxivCategory = Literal[
    "cs.AI", "cs.CL", "cs.LG", "cs.CV", "cs.CR", "cs.DS",
    "quant-ph", "math.OC", "stat.ML", "physics.comp-ph",
    "q-bio.QM", "econ.TH",
]


class SearchArxivArgs(BaseModel):
    """Arguments for an arXiv academic-paper search."""

    query: str = Field(
        ...,
        description=(
            "Search query. Can be a topic ('quantum error correction'), an author name "
            "('Peter Shor'), or a paper title. Supports boolean operators (AND, OR, NOT)."
        ),
    )
    max_results: int = Field(
        default=10, ge=1, le=50,
        description="Number of papers to return.",
    )
    category: Optional[ArxivCategory] = Field(
        default=None,
        description="Restrict search to a specific arXiv subject category. Omit to search all.",
    )
    date_from: Optional[str] = Field(
        default=None,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="Lower bound for submission date, in YYYY-MM-DD format.",
    )
    date_to: Optional[str] = Field(
        default=None,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="Upper bound for submission date, in YYYY-MM-DD format.",
    )
    sort_by: Literal["relevance", "submittedDate", "lastUpdatedDate"] = Field(
        default="relevance",
        description="Ordering. 'relevance' = best textual match; 'submittedDate' = newest first.",
    )


class SearchWikipediaArgs(BaseModel):
    """Arguments for a Wikipedia article summary lookup."""

    query: str = Field(
        ...,
        description=(
            "Article title or topic to look up. Best results come from canonical names "
            "('Albert Einstein', 'Quantum entanglement') rather than questions."
        ),
    )
    language: str = Field(
        default="en",
        pattern=r"^[a-z]{2}$",
        description=(
            "Wikipedia language edition as a 2-letter ISO 639-1 code "
            "(e.g., 'en', 'es', 'hi', 'te')."
        ),
    )
    summary_length: Literal["short", "medium", "full"] = Field(
        default="medium",
        description=(
            "How much of the article to return. 'short' = first paragraph, "
            "'medium' = lead section, 'full' = entire article body."
        ),
    )


class FetchYoutubeTranscriptArgs(BaseModel):
    """Arguments for fetching a YouTube video transcript."""

    video: str = Field(
        ...,
        pattern=r"^(https?://(www\.)?(youtube\.com|youtu\.be)/.+|[A-Za-z0-9_-]{11})$",
        description=(
            "Identifies the video. Accepts either a full YouTube URL "
            "(youtube.com/watch?v=ID, youtu.be/ID, or shorts URL) "
            "OR a bare 11-character video ID (e.g., 'dQw4w9WgXcQ')."
        ),
    )
    language: str = Field(
        default="en",
        pattern=r"^[a-z]{2}$",
        description="Preferred transcript language as a 2-letter ISO 639-1 code.",
    )
    include_timestamps: bool = Field(
        default=False,
        description="If true, prefix each line with its start time in [MM:SS] format.",
    )


class CalculatorArgs(BaseModel):
    """Arguments for a safe math-expression evaluator."""

    expression: str = Field(
        ...,
        max_length=500,
        pattern=r"^[0-9\s+\-*/^%().,a-z_]+$",
        description=(
            "Math expression in standard infix notation. "
            "Examples: '2 + 3 * 4', 'sqrt(144) + log(100)', 'sin(pi/2)'. "
            "Must contain only digits, math operators, parentheses, decimal points, "
            "commas, whitespace, and recognized function/constant names. No code syntax."
        ),
    )
    precision: int = Field(
        default=6, ge=0, le=15,
        description="Number of decimal places in the returned result.",
    )


# ===========================================================================
# SECTION 2 — The "rewrap once" helper
# ===========================================================================

def pydantic_to_openai_tool(
    name: str,
    description: str,
    model: type[BaseModel],
) -> dict:
    """Convert a Pydantic BaseModel into an OpenAI-format tool definition.
    The model's JSON Schema becomes the 'parameters' block.
    """
    schema = model.model_json_schema()
    schema.pop("title", None)
    for prop in schema.get("properties", {}).values():
        prop.pop("title", None)

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": schema,
        },
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
            "Returns titles, authors, abstracts, and arXiv IDs. "
            "Use for academic research, not general web."
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
            "Use only when the user gives a specific video (URL or ID), "
            "not to discover videos on a topic."
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



_EXECUTOR = ThreadPoolExecutor(max_workers=4)

# Pydantic schema (model-facing) uses long names; DDGS API uses short codes.
# Translating between contracts is the tool's job — the model shouldn't have
# to know DDGS's internal vocabulary.
_RECENCY_MAP = {"day": "d", "week": "w", "month": "m", "year": "y"}


def run_search_web(args: SearchWebArgs) -> str:
    """Real DDGS implementation with a hard 8s timeout and error handling."""

    def _search() -> str:
        try:
            ddgs = DDGS()
            results = ddgs.text(
                args.query,
                max_results=args.max_results,
                timelimit=_RECENCY_MAP.get(args.recency) if args.recency else None,
            )
            output = []
            for i, r in enumerate(results, 1):
                output.append(
                    f"{i}. {r.get('title', 'Untitled')}\n"
                    f"   {r.get('body', '')}\n"
                    f"   Source: {r.get('href', 'N/A')}"
                )
            if not output:
                return "[search_web] No results found."
            return "\n".join(output)
        except Exception as e:
            return f"[search_web error] {type(e).__name__}: {e}"

    future = _EXECUTOR.submit(_search)
    try:
        return future.result(timeout=0.001) # 8-second timeout
    except FutureTimeoutError:
        return "[search_web error] Request timed out after 8 seconds."
    except Exception as e:
        return f"[search_web error] {type(e).__name__}: {e}"


def run_search_arxiv(args: SearchArxivArgs) -> str:
    return "[NOT IMPLEMENTED] search_arxiv will be wired in Phase 3."


def run_search_wikipedia(args: SearchWikipediaArgs) -> str:
    return "[NOT IMPLEMENTED] search_wikipedia will be wired in Phase 3."


def run_fetch_youtube_transcript(args: FetchYoutubeTranscriptArgs) -> str:
    return "[NOT IMPLEMENTED] fetch_youtube_transcript will be wired in Phase 3."


def run_calculator(args: CalculatorArgs) -> str:
    """Safe math evaluator using numexpr (no Python code execution path)."""
    try:
        import numexpr
    except ImportError:
        return "[error] numexpr not installed. pip install numexpr"

    try:
        expr = args.expression.replace("^", "**")
        result = numexpr.evaluate(expr)
        return str(round(float(result), args.precision))
    except Exception as e:
        return f"[calculator error] {type(e).__name__}: {e}"


TOOL_DISPATCH = {
    "search_web": run_search_web,
    "search_arxiv": run_search_arxiv,
    "search_wikipedia": run_search_wikipedia,
    "fetch_youtube_transcript": run_fetch_youtube_transcript,
    "calculator": run_calculator,
}


def dispatch_tool(name: str, raw_args: dict) -> str:
    """Validate raw args against the Pydantic schema, then run the tool."""
    if name not in TOOL_MODELS:
        return f"[error] Unknown tool: {name}"
    try:
        validated = TOOL_MODELS[name].model_validate(raw_args)
    except ValidationError as e:
        return f"[validation error] {e.errors()}"
    return TOOL_DISPATCH[name](validated)


# ===========================================================================
# SECTION 5 — Agent loop
# ===========================================================================

def get_system_prompt() -> str:
    """System prompt with date anchor and staleness warning."""
    today = datetime.now().strftime("%Y-%m-%d")
    return (
        f"Today's date is {today}. Your training data cuts off in April 2024 and is stale. "
        f"You MUST call search_web for any time-sensitive query: current events, recent news, "
        f"stock prices, product releases, software versions, or anything that may have changed. "
        f"Pick the most specific tool: search_arxiv for academic papers, search_wikipedia for "
        f"encyclopedic facts, fetch_youtube_transcript for a specific given video, calculator for math, "
        f"search_web for everything else. Call tools one at a time and reason about results before "
        f"deciding the next step."
        f" If a tool returns a string starting with 'ERROR:', do not retry blindly — "
        f"either rephrase the query once and try again, or tell the user the tool failed "
        f"and answer from general knowledge with a clear caveat."
    )


client = OpenAI()


def run_agent(user_message: str, max_turns: int = 5) -> str:
    """Run the agentic loop. max_turns=5 balances cost/quality."""
    messages = [
        {"role": "system", "content": get_system_prompt()},
        {"role": "user", "content": user_message},
    ]

    for turn in range(max_turns):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )
        msg = response.choices[0].message

        if msg.tool_calls:
            messages.append(msg)
            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    raw_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError as e:
                    result = f"[error] Invalid JSON in tool args: {e}"
                else:
                    result = dispatch_tool(name, raw_args)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
            continue

        return msg.content or ""

    return "[agent] Hit max_turns without producing a final answer."


# ===========================================================================
# SECTION 6 — CLI entry
# ===========================================================================

def main() -> None:
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        print(run_agent(question))
        return

    print("Agent ready. Type a question, or 'exit' to quit.\n")
    while True:
        try:
            q = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if q.lower() in {"exit", "quit"}:
            break
        if not q:
            continue
        print(f"agent > {run_agent(q)}\n")


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY in your environment first.", file=sys.stderr)
        sys.exit(1)
    main()