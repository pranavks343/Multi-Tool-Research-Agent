# app/agent/executor.py
"""The agentic loop: send messages, execute tool calls, repeat until a final answer."""
from __future__ import annotations

import json
import sys

from app.memory import ConversationMemory
from app.agent.prompts import get_system_prompt
from app.config import COST_CAP, DEBUG, INPUT_RATE, MAX_TURNS, MODEL, OUTPUT_RATE, client
from app.schemas import error_payload
from app.tools import TOOLS, dispatch_tool


def run_agent(
    user_message: str,
    memory: ConversationMemory | None = None,
    max_turns: int = MAX_TURNS,
    called_tools: list[str] | None = None,
) -> str:
    
    memory = memory or ConversationMemory(
        {"role": "system", "content": get_system_prompt()}
    )
    memory.add({"role": "user", "content": user_message})

    cost = 0.0

    for _turn in range(max_turns):
        if cost >= COST_CAP:
            return f"[agent] Cost cap of ${COST_CAP:.2f} reached (spent ${cost:.4f}). Stopping."

        response = client.chat.completions.create(
            model=MODEL,
            messages=memory.get_messages(),   # trimmed, budget-aware history
            tools=TOOLS,
            tool_choice="auto",
        )

        u = response.usage
        turn_cost = u.prompt_tokens * INPUT_RATE + u.completion_tokens * OUTPUT_RATE
        cost += turn_cost

        if DEBUG:
            print(f"[cost] turn={_turn} +${turn_cost:.5f} "
                  f"total=${cost:.5f}", file=sys.stderr)

        msg = response.choices[0].message

        if msg.tool_calls:
            # Convert the SDK object to a plain dict so ConversationMemory's
            # dict-based logic (.get / token counting) works on it.
            memory.add(msg.model_dump(exclude_none=True))

            # Sequential: one layer of execution, no nested thread pools.
            # Per-tool timeouts still live inside each tool's own _EXECUTOR.
            for tc in msg.tool_calls:
                name = tc.function.name
                if called_tools is not None:
                    called_tools.append(name)
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
        memory.add({"role": "assistant", "content": answer})
        return answer

    return "[agent] Hit max_turns without producing a final answer."