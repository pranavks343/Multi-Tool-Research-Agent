
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from fastapi import FastAPI
from pydantic import BaseModel

from app.agent.executor import run_agent

app = FastAPI(title="Multi-Tool Research Agent", version="0.1.0")


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    tools_used: list[str]


@app.get("/health")
def health() -> dict:
    """Liveness probe — used by Render/Docker to confirm the service is up."""
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    """Run one question through the agent (stateless — no cross-request memory)."""
    called: list[str] = []
    answer = run_agent(req.question, called_tools=called)
    return QueryResponse(answer=answer, tools_used=called)