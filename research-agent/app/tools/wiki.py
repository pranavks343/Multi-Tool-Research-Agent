"""Wikipedia article summary — via the `wikipedia-api` library."""
from __future__ import annotations

import json
from concurrent.futures import TimeoutError as FutureTimeoutError
from datetime import datetime, timezone

from app.config import _EXECUTOR, WIKI_TIMEOUT
from app.schemas import SearchWikipediaArgs, ToolResult, error_payload


def run_search_wikipedia(args: SearchWikipediaArgs) -> str:
    """Return a JSON list[ToolResult] (single item) on success, or error JSON on failure."""
    try:
        import wikipediaapi
    except ImportError:
        return error_payload("search_wikipedia", "wikipedia-api not installed (uv add wikipedia-api)")

    def _search() -> str:
        try:
            fetched_at = datetime.now(timezone.utc)
            # Wikipedia requires a descriptive User-Agent or it returns 403.
            wiki = wikipediaapi.Wikipedia(
                user_agent="ResearchAgent/0.1 (educational)", language=args.language
            )
            page = wiki.page(args.query)
            if not page.exists():
                return error_payload("search_wikipedia", f"No article for '{args.query}'")

            if args.summary_length == "short":
                content = page.summary.split("\n")[0]
            elif args.summary_length == "medium":
                content = page.summary
            else:
                content = page.text[:10000]  # cap "full" to avoid context blowup

            result = ToolResult(
                source="Wikipedia",
                title=page.title,
                snippet=content,
                url=page.fullurl,
                timestamp=fetched_at,
            )
            return json.dumps([result.model_dump(mode="json")])
        except Exception as e:
            return error_payload("search_wikipedia", f"{type(e).__name__}: {e}")

    future = _EXECUTOR.submit(_search)
    try:
        return future.result(timeout=WIKI_TIMEOUT)
    except FutureTimeoutError:
        return error_payload("search_wikipedia", f"Timeout after {WIKI_TIMEOUT}s")
