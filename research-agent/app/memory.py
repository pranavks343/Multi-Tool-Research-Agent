import tiktoken

_ENCODING = tiktoken.encoding_for_model("gpt-4o-mini")


def count_tokens(messages: list[dict]) -> int:
    """Rough token count of a message list. Good enough for budgeting."""
    total = 0
    for m in messages:
        content = m.get("content") or ""
        if not isinstance(content, str):
            content = str(content)
        total += len(_ENCODING.encode(content)) + 4
    return total


class ConversationMemory:
    def __init__(self, system_message: dict):
        self._messages: list[dict] = [system_message]

    def add(self, message: dict) -> None:
        self._messages.append(message)

    def get_messages(self, token_budget: int = 8000) -> list[dict]:
        system, rest = self._messages[0], self._messages[1:]

        
        groups: list[list[dict]] = []
        for m in rest:
            role = m.get("role")
            if role == "tool" and groups and self._is_open_tool_group(groups[-1]):
                groups[-1].append(m)          
            elif role == "assistant" and m.get("tool_calls"):
                groups.append([m])            
            else:
                groups.append([m])            

        
        budget = token_budget - count_tokens([system])
        kept: list[list[dict]] = []
        running = 0
        for group in reversed(groups):    #we start from the end because the newest context matters more
            cost = count_tokens(group)
            if running + cost > budget:
                break                          
            kept.append(group)
            running += cost

        kept.reverse()
        return [system] + [m for g in kept for m in g]

    @staticmethod
    def _is_open_tool_group(group: list[dict]) -> bool:
        """True if the group started with an assistant tool-call (so trailing
        tool messages belong to it)."""
        head = group[0]
        return head.get("role") == "assistant" and bool(head.get("tool_calls"))