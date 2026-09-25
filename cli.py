"""Run one brief in the terminal.

    python cli.py "Research LangGraph and tell me what it does."
"""

from __future__ import annotations

import sys

from folio.agent import stream_research


def main() -> int:
    query = " ".join(sys.argv[1:]).strip() or "Research LangGraph and tell me what it does."
    exit_code = 0
    for event in stream_research(query):
        kind = event.get("type")
        if kind == "step":
            trace = event.get("trace") or {}
            ms = trace.get("ms")
            suffix = f"  ({ms} ms)" if isinstance(ms, int) else ""
            print(f"→ {trace.get('title', event.get('node'))}: {trace.get('message', '')}{suffix}")
        elif kind == "answer":
            print()
            print(event["answer"]["plain"])
        elif kind == "done":
            print()
            print(f"— {event['ms']} ms")
        elif kind == "error":
            print(event.get("message", "error"), file=sys.stderr)
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
