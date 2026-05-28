"""Pydantic models: the model-facing argument schemas and the ToolResult envelope.

`schemas.py` is the single source of truth for every tool's input contract and the
shared output shape. It has no internal dependencies beyond pydantic.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field



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


def error_payload(tool: str, message: str) -> str:
    """Uniform error JSON every tool returns on failure.

    Shared here (rather than duplicated per tool) so the error contract is defined
    in exactly one place — the same module that defines the success contract.
    """
    return json.dumps({"error": message, "tool": tool})
