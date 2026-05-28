"""Web search tool — DuckDuckGo via the `ddgs` library."""
from __future__ import annotations

import json
from concurrent.futures import TimeoutError as FutureTimeoutError
from datetime import datetime, timezone

from ddgs import DDGS

from app.config import _EXECUTOR, _RECENCY_MAP, SEARCH_TIMEOUT
from app.schemas import SearchWebArgs, ToolResult, error_payload


def run_search_web(args: SearchWebArgs) -> str:
    """Return a JSON list[ToolResult] on success, or error JSON on failure. Hard timeout."""

    def _search() -> str:
        try:
            fetched_at = datetime.now(timezone.utc)
            ddgs = DDGS()
            raw_results = ddgs.text(
                args.query,
                max_results=args.max_results,
                timelimit=_RECENCY_MAP.get(args.recency) if args.recency else None,
            )
            results = [
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
            return error_payload("search_web", f"{type(e).__name__}: {e}")

    future = _EXECUTOR.submit(_search)
    try:
        return future.result(timeout=SEARCH_TIMEOUT)
    except FutureTimeoutError:
        return error_payload("search_web", f"Request timed out after {SEARCH_TIMEOUT}s")
    except Exception as e:
        return error_payload("search_web", f"{type(e).__name__}: {e}")
