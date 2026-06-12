from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

# Make `app` importable when run as a script from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent.executor import run_agent  # noqa: E402

DATASET_PATH = Path(__file__).resolve().parent / "dataset.jsonl"

# Tools that are infrastructure, not research capabilities. A call to one of these
# must NOT count toward (or against) a "none" expectation. Empty today — this
# codebase has no model-callable bookkeeping tools — but encoding the filter now
# means adding e.g. a `log_event` tool later won't silently break "none" scoring.
BOOKKEEPING_TOOLS: set[str] = set()

# The full set of real research tools, used to validate the dataset.
RESEARCH_TOOLS = {
    "search_web", "search_arxiv", "search_wikipedia",
    "fetch_youtube_transcript", "calculator",
}


def _normalize(expected) -> list[str]:
    """Always-a-list. 'search_web' -> ['search_web']; lists pass through."""
    return expected if isinstance(expected, list) else [expected]


def _validate_row(row: dict, line_no: int) -> list[str]:
    """Reject malformed labels loudly instead of letting them inflate the score.

    The key rule: 'none' must be a standalone expectation. Mixing 'none' with a
    real tool (['none', 'search_web']) makes every outcome 'correct' — the metric
    becomes unfalsifiable. Fail fast on that.
    """
    expected = _normalize(row["expected_tool"])
    if "none" in expected and len(expected) > 1:
        raise ValueError(
            f"Line {line_no}: 'none' cannot be combined with real tools "
            f"({expected}). 'none' must stand alone or the metric is meaningless."
        )
    for t in expected:
        if t != "none" and t not in RESEARCH_TOOLS:
            raise ValueError(f"Line {line_no}: unknown tool '{t}' in {expected}")
    return expected


def score_one(expected: list[str], called_tools: list[str]) -> bool:
    """Apply the agreed scoring rule.

    - 'none'  -> correct iff no research tool fired.
    - else    -> correct iff ANY expected tool is present among the calls
                 ('present among', not exclusivity — a comparison question may
                 legitimately fan out to several tools).
    """
    research_calls = [t for t in called_tools if t not in BOOKKEEPING_TOOLS]
    if expected == ["none"]:
        return len(research_calls) == 0
    return any(t in research_calls for t in expected)


def main() -> None:
    rows = []
    with DATASET_PATH.open() as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["_expected"] = _validate_row(row, i)  # validate up front
            rows.append(row)

    total = len(rows)
    correct = 0
    per_tool_total: dict[str, int] = defaultdict(int)
    per_tool_correct: dict[str, int] = defaultdict(int)
    failures = []

    print(f"Running {total} eval questions...\n")

    for idx, row in enumerate(rows, 1):
        question = row["question"]
        expected = row["_expected"]
        label = "|".join(expected)  # for per-tool grouping

        called: list[str] = []
        try:
            run_agent(question, called_tools=called)
        except Exception as e:
            called = [f"<error: {type(e).__name__}>"]

        ok = score_one(expected, called)
        correct += ok
        per_tool_total[label] += 1
        per_tool_correct[label] += ok

        mark = "PASS" if ok else "FAIL"
        print(f"[{idx:2d}/{total}] {mark}  expected={label:<35} called={called}")
        if not ok:
            failures.append((question, expected, called))

    accuracy = correct / total if total else 0.0
    print("\n" + "=" * 60)
    print(f"TOOL-SELECTION ACCURACY: {correct}/{total} = {accuracy:.1%}")
    print("=" * 60)

    print("\nPer-expectation breakdown:")
    for label in sorted(per_tool_total):
        c, t = per_tool_correct[label], per_tool_total[label]
        print(f"  {label:<40} {c}/{t} = {c / t:.0%}")

    if failures:
        print(f"\n{len(failures)} failure(s) — where routing went wrong:")
        for q, exp, got in failures:
            print(f"  Q: {q}")
            print(f"     expected one of {exp}, agent called {got}")


if __name__ == "__main__":
    main()