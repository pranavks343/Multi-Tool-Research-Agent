"""Shared configuration: constants, the thread-pool executor, and the OpenAI client.

This module has NO internal dependencies — everything else imports from here, so it
must stay at the bottom of the dependency graph.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI

# Load .env BEFORE constructing the OpenAI client so OPENAI_API_KEY is available.
load_dotenv(find_dotenv())

# --- Model / loop ---
MODEL = "gpt-4o-mini"
MAX_TURNS = 5

# --- Per-tool timeouts (seconds) ---
# Named constants instead of magic numbers scattered across tool files.
SEARCH_TIMEOUT = 8.0
ARXIV_TIMEOUT = 15.0   # arXiv's API is noticeably slower
WIKI_TIMEOUT = 8.0
YOUTUBE_TIMEOUT = 15.0

# --- Shared thread pool for hard tool timeouts ---
# Module-level (no `with` block) so future.result(timeout=...) is a real wall-clock
# ceiling: when it raises, we return immediately and the orphaned worker dies with
# the process instead of blocking shutdown(wait=True). Threads can't be killed in
# CPython — for a true hard kill of untrusted code you'd need a subprocess.
_EXECUTOR = ThreadPoolExecutor(max_workers=4)

# DuckDuckGo's `timelimit` arg uses short codes; the model-facing schema uses long
# names. Translating between the two contracts is the tool's job, not the model's.
_RECENCY_MAP = {"day": "d", "week": "w", "month": "m", "year": "y"}

# Toggle the [TOOL DEBUG] stderr trace. Set RESEARCH_AGENT_DEBUG=0 to silence it.
DEBUG = os.getenv("RESEARCH_AGENT_DEBUG", "1") == "1"

# Single shared client (constructed once, reused for every call).
client = OpenAI()
