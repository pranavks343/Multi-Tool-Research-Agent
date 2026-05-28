"""CLI entry point for the multi-tool research agent.

Run from the project root:
    uv run scripts/agent_cli.py "your question"
    uv run scripts/agent_cli.py            # interactive REPL
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the project root importable so `from app...` resolves no matter the CWD.
# (Running a script puts scripts/ on sys.path, not the project root, so we add it.)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent.executor import run_agent  # noqa: E402  (import after sys.path tweak)


def main() -> None:
    if len(sys.argv) > 1:
        print(run_agent(" ".join(sys.argv[1:])))
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
    main()
