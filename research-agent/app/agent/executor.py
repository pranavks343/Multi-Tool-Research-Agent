# app/agent/executor.py
"""The agentic loop: send messages, execute tool calls, repeat until a final answer."""
from __future__ import annotations

import json

from app.memory import ConversationMemory
from app.agent.prompts import get_system_prompt
from app.config import MAX_TURNS, MODEL, client
from app.schemas import error_payload
from app.tools import TOOLS, dispatch_tool


def run_agent(
    user_message: str,
    memory: ConversationMemory | None = None,
    max_turns: int = MAX_TURNS,
) -> str:
    # No memory injected → fresh, throwaway instance (single-shot).
    # Memory injected → reused across calls (REPL persists).
    memory = memory or ConversationMemory(
        {"role": "system", "content": get_system_prompt()}
    )
    memory.add({"role": "user", "content": user_message})

    for _turn in range(max_turns):
        response = client.chat.completions.create(
            model=MODEL,
            messages=memory.get_messages(),   # trimmed, budget-aware history
            tools=TOOLS,
            tool_choice="auto",
        )
        msg = response.choices[0].message

        if msg.tool_calls:
            # Convert the SDK object to a plain dict so ConversationMemory's
            # dict-based logic (.get / token counting) works on it.
            memory.add(msg.model_dump(exclude_none=True))
            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    raw_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError as e:
                    result = error_payload(name, f"Invalid JSON in tool args: {e}")
                else:
                    result = dispatch_tool(name, raw_args)

                memory.add({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
            continue

        answer = msg.content or ""
        memory.add({"role": "assistant", "content": answer})  # persist the turn
        return answer

    return "[agent] Hit max_turns without producing a final answer."