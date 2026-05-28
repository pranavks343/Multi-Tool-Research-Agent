"""arXiv academic-paper search — via the `arxiv` library."""
from __future__ import annotations

import json
from concurrent.futures import TimeoutError as FutureTimeoutError
from datetime import datetime, timezone

from app.config import _EXECUTOR, ARXIV_TIMEOUT
from app.schemas import SearchArxivArgs, ToolResult, error_payload


def run_search_arxiv(args: SearchArxivArgs) -> str:
    """Return a JSON list[ToolResult] on success, or error JSON on failure."""
    try:
        import arxiv
    except ImportError:
        return error_payload("search_arxiv", "arxiv not installed (uv add arxiv)")

    def _search() -> str:
        try:
            fetched_at = datetime.now(timezone.utc)
            # arXiv quirk: category filtering is done via a `cat:` prefix in the query.
            query = f"cat:{args.category} AND ({args.query})" if args.category else args.query
            sort_map = {
                "relevance": arxiv.SortCriterion.Relevance,
                "submittedDate": arxiv.SortCriterion.SubmittedDate,
                "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
            }
            client = arxiv.Client()
            search = arxiv.Search(
                query=query, max_results=args.max_results, sort_by=sort_map[args.sort_by]
            )

            results = []
            for paper in client.results(search):
                # arXiv API has no date-range filter, so we filter client-side.
                pub = paper.published.strftime("%Y-%m-%d")
                if args.date_from and pub < args.date_from:
                    continue
                if args.date_to and pub > args.date_to:
                    continue
                authors = ", ".join(a.name for a in paper.authors[:3])
                if len(paper.authors) > 3:
                    authors += ", et al."
                results.append(ToolResult(
                    source="arXiv",
                    title=paper.title,
                    snippet=f"{authors} ({paper.published.year}). {paper.summary[:400]}",
                    url=paper.entry_id,
                    timestamp=fetched_at,
                ))
            if not results:
                return error_payload("search_arxiv", "No results found.")
            return json.dumps([r.model_dump(mode="json") for r in results])
        except Exception as e:
            return error_payload("search_arxiv", f"{type(e).__name__}: {e}")

    future = _EXECUTOR.submit(_search)
    try:
        return future.result(timeout=ARXIV_TIMEOUT)
    except FutureTimeoutError:
        return error_payload("search_arxiv", f"Timeout after {ARXIV_TIMEOUT}s")
