"""Safe math evaluator — via `numexpr` (no Python code-execution path)."""
from __future__ import annotations

from app.schemas import CalculatorArgs


def run_calculator(args: CalculatorArgs) -> str:
    """Return the numeric result as a string, or an error string on failure.

    Unlike the retrieval tools, the calculator returns a bare value (not a
    ToolResult envelope) because there is no source/url/timestamp to attach.
    numexpr is the safe evaluator here — it parses a restricted math grammar and
    never executes arbitrary Python (no eval()).
    """
    try:
        import numexpr
    except ImportError:
        return "[error] numexpr not installed (uv add numexpr)"
    try:
        expr = args.expression.replace("^", "**")  # treat ^ as exponent, not XOR
        result = numexpr.evaluate(expr)
        return str(round(float(result), args.precision))
    except Exception as e:
        return f"[calculator error] {type(e).__name__}: {e}"
