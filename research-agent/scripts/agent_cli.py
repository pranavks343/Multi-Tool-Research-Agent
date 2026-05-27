from __future__ import annotations
import json
import os
import sys
from datetime import datetime, timezone
from typing import Literal, Optional
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

from ddgs import DDGS
load_dotenv(find_dotenv())


class SearchWebArgs(BaseModel):
    query: str = Field(..., description="The search query. Free-form natural language.")
    max_results: int = Field(default=5, ge=1, le=20, description="Number of results to return.")
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
    query: str = Field(
        ...,
        description=(
            "Search query. Can be a topic ('quantum error correction'), an author name "
            "('Peter Shor'), or a paper title. Supports boolean operators (AND, OR, NOT)."
        ),
    )
    max_results: int = Field(default=10, ge=1, le=50, description="Number of papers to return.")
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
        description="Wikipedia language edition as a 2-letter ISO 639-1 code (e.g., 'en', 'es', 'hi', 'te').",
    )
    summary_length: Literal["short", "medium", "full"] = Field(
        default="medium",
        description=(
            "How much of the article to return. 'short' = first paragraph, "
            "'medium' = lead section, 'full' = entire article body."
        ),
    )


class FetchYoutubeTranscriptArgs(BaseModel):
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


class ToolResult(BaseModel):
    title: str = Field(..., description="Headline or display name of the hit.")
    snippet: str = Field(..., description="Body/description/excerpt text.")
    url: str = Field(..., description="Source URL — used by the LLM as the inline citation anchor.")
    source: str = Field(..., description="Provider name (e.g., 'DuckDuckGo', 'arXiv', 'Wikipedia', 'YouTube').")
    timestamp: datetime = Field(..., description="When this result was retrieved (ISO 8601, UTC).")


def pydantic_to_openai_tool(name: str, description: str, model: type[BaseModel]) -> dict:
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


_EXECUTOR = ThreadPoolExecutor(max_workers=4)
_RECENCY_MAP = {"day": "d", "week": "w", "month": "m", "year": "y"}


def _error_payload(tool: str, message: str) -> str:
    return json.dumps({"error": message, "tool": tool})


def run_search_web(args: SearchWebArgs) -> str:
    def _search() -> str:
        try:
            fetched_at = datetime.now(timezone.utc)
            ddgs = DDGS()
            raw_results = ddgs.text(
                args.query,
                max_results=args.max_results,
                timelimit=_RECENCY_MAP.get(args.recency) if args.recency else None,
            )
            results: list[ToolResult] = [
                ToolResult(
                    title=r.get("title") or "N/A",
                    snippet=r.get("body") or "N/A",
                    url=r.get("href") or "N/A",
                    source="DuckDuckGo",
                    timestamp=fetched_at,
                )
                for r in raw_results
            ]
            return json.dumps([r.model_dump(mode="json") for r in results])
        except Exception as e:
            return _error_payload("search_web", f"{type(e).__name__}: {e}")

    future = _EXECUTOR.submit(_search)
    try:
        return future.result(timeout=8.0)
    except FutureTimeoutError:
        return _error_payload("search_web", "Request timed out after 8 seconds.")
    except Exception as e:
        return _error_payload("search_web", f"{type(e).__name__}: {e}")


def run_search_arxiv(args: SearchArxivArgs) -> str:
    """arXiv search via the `arxiv` library. Returns JSON list[ToolResult]."""
    try:
        import arxiv
    except ImportError:
        return json.dumps({"error": "arxiv not installed", "tool": "search_arxiv"})

    def _search():
        try:
            fetched_at = datetime.now()
            query = f"cat:{args.category} AND ({args.query})" if args.category else args.query
            sort_map = {
                "relevance": arxiv.SortCriterion.Relevance,
                "submittedDate": arxiv.SortCriterion.SubmittedDate,
                "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
            }
            client = arxiv.Client()
            search = arxiv.Search(query=query, max_results=args.max_results, sort_by=sort_map[args.sort_by])

            results = []
            for paper in client.results(search):
                pub = paper.published.strftime("%Y-%m-%d")
                if args.date_from and pub < args.date_from: continue
                if args.date_to and pub > args.date_to: continue
                authors = ", ".join(a.name for a in paper.authors[:3])
                if len(paper.authors) > 3: authors += ", et al."
                results.append(ToolResult(
                    title=paper.title,
                    snippet=f"{authors} ({paper.published.year}). {paper.summary[:400]}",
                    url=paper.entry_id,
                    timestamp=fetched_at,
                ))
            if not results:
                return json.dumps({"error": "No results found.", "tool": "search_arxiv"})
            return json.dumps([r.model_dump(mode="json") for r in results])
        except Exception as e:
            return json.dumps({"error": f"{type(e).__name__}: {e}", "tool": "search_arxiv"})

    future = _EXECUTOR.submit(_search)
    try:
        return future.result(timeout=15.0)  # arxiv API is slow
    except FutureTimeoutError:
        return json.dumps({"error": "Timeout after 15s", "tool": "search_arxiv"})


def run_search_wikipedia(args: SearchWikipediaArgs) -> str:
    """Wikipedia summary via `wikipedia-api`. Returns JSON list[ToolResult]."""
    try:
        import wikipediaapi
    except ImportError:
        return json.dumps({"error": "wikipedia-api not installed", "tool": "search_wikipedia"})

    def _search():
        try:
            fetched_at = datetime.now()
            wiki = wikipediaapi.Wikipedia(
                user_agent="ResearchAgent/0.1 (educational)",
                language=args.language,
            )
            page = wiki.page(args.query)
            if not page.exists():
                return json.dumps({"error": f"No article for '{args.query}'", "tool": "search_wikipedia"})

            if args.summary_length == "short":
                content = page.summary.split("\n")[0]
            elif args.summary_length == "medium":
                content = page.summary
            else:
                content = page.text[:10000]  # cap to avoid context blowup

            result = ToolResult(
                title=page.title,
                snippet=content,
                url=page.fullurl,
                timestamp=fetched_at,
            )
            return json.dumps([result.model_dump(mode="json")])
        except Exception as e:
            return json.dumps({"error": f"{type(e).__name__}: {e}", "tool": "search_wikipedia"})

    future = _EXECUTOR.submit(_search)
    try:
        return future.result(timeout=8.0)
    except FutureTimeoutError:
        return json.dumps({"error": "Timeout after 8s", "tool": "search_wikipedia"})


def run_fetch_youtube_transcript(args: FetchYoutubeTranscriptArgs) -> str:
    """YouTube transcript via `youtube-transcript-api`. Returns JSON list[ToolResult]."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError:
        return json.dumps({"error": "youtube-transcript-api not installed", "tool": "fetch_youtube_transcript"})

    def _extract_id(video: str) -> str:
        if len(video) == 11 and "/" not in video and "." not in video:
            return video
        import re
        m = re.search(r"(?:v=|youtu\.be/|/shorts/|/embed/)([A-Za-z0-9_-]{11})", video)
        if m: return m.group(1)
        raise ValueError(f"Could not extract video ID from: {video}")

    def _fetch():
        try:
            fetched_at = datetime.now()
            video_id = _extract_id(args.video)
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[args.language])

            if args.include_timestamps:
                content = "\n".join(
                    f"[{int(t['start']//60):02d}:{int(t['start']%60):02d}] {t['text']}"
                    for t in transcript
                )
            else:
                content = " ".join(t["text"] for t in transcript)

            if len(content) > 15000:
                content = content[:15000] + " ... [truncated]"

            result = ToolResult(
                title=f"YouTube video {video_id}",
                snippet=content,
                url=f"https://youtube.com/watch?v={video_id}",
                timestamp=fetched_at,
            )
            return json.dumps([result.model_dump(mode="json")])
        except Exception as e:
            return json.dumps({"error": f"{type(e).__name__}: {e}", "tool": "fetch_youtube_transcript"})

    future = _EXECUTOR.submit(_fetch)
    try:
        return future.result(timeout=15.0)
    except FutureTimeoutError:
        return json.dumps({"error": "Timeout after 15s", "tool": "fetch_youtube_transcript"})


def run_calculator(args: CalculatorArgs) -> str:
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
    if name not in TOOL_MODELS:
        return _error_payload(name, f"Unknown tool: {name}")
    try:
        validated = TOOL_MODELS[name].model_validate(raw_args)
    except ValidationError as e:
        return _error_payload(name, f"validation error: {e.errors()}")
    return TOOL_DISPATCH[name](validated)


def get_system_prompt() -> str:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return (
        f"Today's date is {today} (UTC). Your training data cuts off in April 2024 and is stale. "
        f"You MUST call search_web for any time-sensitive query: current events, recent news, "
        f"stock prices, product releases, software versions, or anything that may have changed. "
        f"Pick the most specific tool: search_arxiv for academic papers, search_wikipedia for "
        f"encyclopedic facts, fetch_youtube_transcript for a specific given video, calculator for math, "
        f"search_web for everything else. Call tools one at a time and reason about results before "
        f"deciding the next step.\n\n"
        f"TOOL OUTPUT CONTRACT:\n"
        f"Retrieval tools (search_web, search_arxiv, search_wikipedia, fetch_youtube_transcript) return a JSON array of "
        f"ToolResult objects. Each object has: 'title', 'snippet', 'url', 'source', 'timestamp'. "
        f"The 'url' field is the citation anchor — you MUST cite every factual claim inline using "
        f"the matching url, in the form (source: <url>) immediately after the claim. "
        f"The 'source' field names the provider (e.g., DuckDuckGo, arXiv, Wikipedia) — use it when "
        f"attributing or comparing across providers. "
        f"The 'timestamp' field is ISO 8601 UTC and tells you when the data was fetched — use it to "
        f"judge freshness for 'latest' / 'recent' queries.\n\n"
        f"THREE RESPONSE SHAPES TO HANDLE:\n"
        f"1. JSON array with items — normal success. Cite using url fields.\n"
        f"2. JSON empty array [] — query ran but found nothing. Tell the user no results were "
        f"found and suggest a refined query. Do NOT retry the same query.\n"
        f"3. JSON object with 'error' and 'tool' fields — the tool failed. You may rephrase the "
        f"query once and retry, or tell the user the tool failed and answer from general knowledge "
        f"with a clear caveat.\n"
        f"Stub tools currently return strings starting with '[NOT IMPLEMENTED]' — treat those as errors."
    )


client = OpenAI()


def run_agent(user_message: str, max_turns: int = 5) -> str:
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
                    result = _error_payload(name, f"Invalid JSON in tool args: {e}")
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