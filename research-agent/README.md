# Multi-Tool Research Agent

A sophisticated research agent that leverages multiple tools to answer complex queries. This agent implements a tool-calling loop that automatically selects the most appropriate information source(s) for each query.

## Features

- **Multi-source Research**: Integrates DuckDuckGo, arXiv, Wikipedia, YouTube, and sandboxed calculation
- **Smart Tool Selection**: Uses LLM to select appropriate tools for each query
- **Response Caching**: 24-hour TTL SQLite cache to reduce API calls
- **Token-Budgeted Memory**: Efficient context window management
- **FastAPI Integration**: RESTful API for query processing
- **Evaluation Framework**: Tool accuracy and hallucination detection metrics

## Project Structure

```
research-agent/
├── app/
│   ├── agent/
│   │   ├── executor.py          # Tool-calling loop (Phase 2 core)
│   │   └── prompts.py           # System prompt and few-shot examples
│   ├── tools/
│   │   ├── web.py               # DuckDuckGo search
│   │   ├── arxiv.py             # arXiv API
│   │   ├── wiki.py              # Wikipedia API
│   │   ├── youtube.py           # YouTube transcript
│   │   └── calc.py              # Sandboxed calculator
│   ├── schemas.py               # Pydantic models
│   ├── memory.py                # Token-budgeted window memory
│   ├── cache.py                 # SQLite response cache
│   ├── config.py                # Configuration
│   ├── api/
│   │   └── main.py              # FastAPI app
│   └── jobs/
│       └── daily_brief.py       # Cron entry point
├── eval/
│   ├── dataset.jsonl            # Evaluation dataset
│   └── tool_accuracy.py         # Evaluation metrics
├── tests/
│   ├── test_tools.py
│   └── test_executor.py
├── scripts/
│   └── agent_cli.py             # CLI entry point
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── .env.example
└── README.md
```

## Quick Start

### 1. Clone and Setup

```bash
cd research-agent
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your API keys
```

### 3. Run CLI (Phase 2 Entry Point)

```bash
python scripts/agent_cli.py "Your research query here"
```

### 4. Start FastAPI Server

```bash
uvicorn app.api.main:app --reload
```

Visit `http://localhost:8000/docs` for API documentation.

## Docker

Build and run with Docker:

```bash
docker-compose up --build
```

## Running Tests

```bash
pytest tests/ -v
```

## Configuration

Key environment variables in `.env`:
- `OPENAI_API_KEY`: Your OpenAI API key
- `LANGSMITH_API_KEY`: Optional LangSmith key for monitoring

See `app/config.py` for additional configuration options including cost caps and iteration limits.

## API Endpoints

### POST /query
Submit a research query.

**Request:**
```json
{
  "query": "What are recent advances in quantum computing?"
}
```

**Response:**
```json
{
  "query": "What are recent advances in quantum computing?",
  "result": "...",
  "tools_used": ["arxiv", "web"],
  "total_tokens": 2500
}
```

## Evaluation

Run evaluation on the labeled dataset:

```bash
python eval/tool_accuracy.py
```

Metrics include:
- Tool selection accuracy
- Hallucination detection rate
- Response quality scores

## Development

### Code Style
- Format: `black` (line-length: 100)
- Linting: `ruff`
- Type hints recommended

### Adding New Tools

1. Create a new module in `app/tools/`
2. Implement tool interface from `schemas.py`
3. Register in agent executor
4. Add tests in `tests/test_tools.py`

## License

MIT
