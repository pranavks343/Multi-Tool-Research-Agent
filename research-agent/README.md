# Multi-Tool Research Agent

An LLM agent that picks the right tool (web, arXiv, Wikipedia, YouTube, calculator)
for each question and returns a source-cited answer.

> Built a multi-tool research agent (5 tools) with 96% tool-selection accuracy on a
> 25-question labeled eval set; token-budgeted conversation memory, per-request cost
> caps, SQLite response caching, and graceful degradation on tool failure.

## Why this project

Most "ask an LLM" systems answer from frozen, stale training data — they can't tell you
today's news, a paper from last week, or the exact result of a calculation. This project
solves that by letting the model *decide* which external tool to reach for on each
question, then grounding its answer in what those tools actually return. That decision —
the model choosing a tool, reading the result, and deciding what to do next — is the
foundation of agentic AI: the difference between a chatbot that guesses and a system that
goes and finds out.

## Architecture

A single question flows through the system like this:

```
User (CLI / REPL / HTTP API)
│
▼
ConversationMemory  ── token-budgeted sliding window, keeps system prompt
│
▼
Agent loop (executor.py)
│  sends trimmed history + tool schemas to the model
▼
Model decides ──── no tool needed ────▶ final answer
│
│ wants a tool
▼
dispatch_tool ── validate args (Pydantic) ── check cache ── run tool
│
▼
Tool (web / arxiv / wiki / youtube / calc)
│  returns ToolResult envelope {title, snippet, url, source, timestamp}
▼
Result fed back into the loop ──┐
│                       │
└──── model decides again (loop until final answer or cost/turn cap)
▼
Cited answer → printed / returned
```

The key insight is in that last arrow: a tool result is **not** the end of the road. It
gets fed back into the model, which then decides whether the answer is complete, whether
to call another tool, or whether to rephrase and retry. The loop repeats — bounded by a
hard turn cap (`MAX_TURNS = 5`) and a per-request cost cap — until the model produces a
final answer. That repetition, the model steering its own next step from real evidence, is
exactly what makes this "agentic" rather than a single canned tool call.

## Evaluation

Tool-selection accuracy on a 25-question labeled set (`eval/dataset.jsonl`):

```
TOOL-SELECTION ACCURACY: 24/25 = 96.0%
calculator                  5/5 = 100%
search_web                  5/5 = 100%
search_arxiv                5/5 = 100%
search_wikipedia            4/4 = 100%
fetch_youtube_transcript    3/3 = 100%
none                        1/2 =  50%
```

The eval earned its keep by finding bugs in my own spec, not just in the model. It caught
a real calculator bug — a `numexpr` `KeyError` when the expression referenced `pi` — and
it surfaced a genuine labeling ambiguity: for "capital of France," should the agent answer
from parametric recall (`none`) or ground the fact in Wikipedia? Both are defensible, which
is why that row scores 50%. An eval that only confirms what you already believe is
theater; one that finds bugs in your *own* spec and ground truth is doing the actual job —
it tells you where your design and your labels disagree with reality.

Run it yourself:

```bash
uv run eval/tool_accuracy.py
```

## Features

- **5 tools**: web search (DuckDuckGo), arXiv, Wikipedia, YouTube transcripts, a sandboxed calculator (numexpr — no `eval`)
- **Token-budgeted memory**: sliding window that respects tool-call message pairing and never drops the system prompt
- **Cost caps**: hard per-request ceiling ($0.05) tracked from real token usage (gpt-4o-mini rates)
- **SQLite caching**: 24h TTL, keyed by validated args, never caches errors
- **Graceful degradation**: tool failures return structured errors; the agent falls back to another tool or answers with a caveat
- **Inline citations**: every factual claim cites the source URL from the ToolResult envelope

## Setup

```bash
# 1. Install dependencies
uv sync

# 2. Add your API key
echo "OPENAI_API_KEY=sk-..." > .env

# 3. Run it
uv run scripts/agent_cli.py "What's the latest news on AI regulation?"   # single-shot
uv run scripts/agent_cli.py                                              # interactive REPL
uv run uvicorn app.api.main:app --port 8000                              # HTTP API
uv run eval/tool_accuracy.py                                             # eval harness
```

### Docker

```bash
docker build -t research-agent .
docker run -p 8000:8000 --env-file .env research-agent
```

### API

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is 2 + 2?"}'
```

## Project structure

```
app/
  config.py        # constants, executor, OpenAI client
  schemas.py       # Pydantic arg models + ToolResult envelope
  memory.py        # token-budgeted ConversationMemory
  cache.py         # SQLite response cache (24h TTL)
  tools/           # one file per tool + registry
  agent/           # prompts + the agentic loop
  api/main.py      # FastAPI wrapper
eval/              # labeled dataset + accuracy harness
scripts/agent_cli.py  # CLI / REPL entry point
```

## Extensions / roadmap

- Token streaming (time-to-first-token)
- Session-scoped memory over HTTP (currently `/query` is stateless)
- Per-tool rate limiting
- Parallel execution of independent tool calls (tool calls within a turn currently run sequentially)
- Differentiated cache TTLs (short for news, long for Wikipedia)
- LangSmith tracing + CI-gated eval

## License

MIT
