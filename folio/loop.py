"""Tool-calling loop.

The fixed graph calls search, then read, then summarize.
This graph does not. A caller node returns one functionCall.
A tools node runs it and sends back a functionResponse.
Then the caller decides again.

No model key is required. The caller follows a short policy:
search, then fetch the best pages one at a time, then summarize.
The objects are the same shape Gemini's function-calling API uses.
"""

from __future__ import annotations

import operator
import time
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from folio.agent import (
    _compose_answer,
    _empty_answer,
    _page_from_hit,
    _reading_list,
    interpret,
    summarize_node,
)
from folio.calling import function_call, function_response
from folio.tools import fetch_page, search_web

MAX_CALLS = 8


class LoopState(TypedDict, total=False):
    query: str
    topic: str
    intent: str
    rationale: str
    search_queries: list[str]
    query_index: int
    hits: list[dict]
    pages: list[dict]
    fetch_index: int
    summary: dict
    answer: dict
    pending: dict
    calls_made: int
    trace: Annotated[list[dict], operator.add]


def caller_node(state: LoopState) -> dict:
    started = time.perf_counter()
    calls_made = state.get("calls_made") or 0
    if calls_made >= MAX_CALLS:
        return _stop(started, "Stopped: too many tool calls.")

    if not state.get("search_queries"):
        plan = interpret(state["query"])
        rationale = (
            f"Tool loop on {plan['topic']}. "
            "The caller will request one tool at a time: search, then fetch, then summarize."
        )
        pending = function_call(
            "search_web",
            {"query": plan["queries"][0], "max_results": 5},
            "call-1",
        )
        return {
            "topic": plan["topic"],
            "intent": plan["intent"],
            "search_queries": plan["queries"],
            "query_index": 0,
            "rationale": rationale,
            "pending": pending,
            "trace": [{
                "node": "agent",
                "title": "Caller",
                "message": rationale,
                "topic": plan["topic"],
                "intent": plan["intent"],
                "queries": plan["queries"],
                "function_call": pending,
                "ms": int((time.perf_counter() - started) * 1000),
            }],
        }

    queries = state.get("search_queries") or []
    query_index = state.get("query_index") or 0
    if query_index < len(queries):
        pending = function_call(
            "search_web",
            {"query": queries[query_index], "max_results": 5},
            f"call-{calls_made + 1}",
        )
        return _ask(started, pending, f"Caller asked search_web for “{queries[query_index]}”.")

    targets = _reading_list(state.get("hits") or [], limit=3)
    fetch_index = state.get("fetch_index") or 0
    if targets and fetch_index < len(targets):
        url = targets[fetch_index]["url"]
        pending = function_call("fetch_page", {"url": url}, f"call-{calls_made + 1}")
        return _ask(started, pending, f"Caller asked to read {url}")

    if state.get("pages") and not state.get("summary"):
        pending = function_call(
            "summarize",
            {"query": state["query"], "documents": len(state["pages"])},
            f"call-{calls_made + 1}",
        )
        return _ask(started, pending, "Caller asked summarize() for a sourced brief.")

    return _stop(started, "Caller has enough to answer.")


def tools_node(state: LoopState) -> dict:
    started = time.perf_counter()
    pending = state.get("pending") or {}
    name = pending.get("name")
    args = pending.get("args") or {}
    call_id = pending.get("id") or "call"
    calls_made = (state.get("calls_made") or 0) + 1

    if name == "search_web":
        return _run_search(state, args, call_id, calls_made, started)
    if name == "fetch_page":
        return _run_fetch(state, args, call_id, calls_made, started)
    if name == "summarize":
        result = summarize_node(state)
        response = function_response(name, call_id, {"claims": (result.get("summary") or {}).get("claims") or 0})
        trace = result.get("trace") or [{}]
        trace[0]["function_response"] = response
        trace[0]["calls"] = [{
            "tool": "summarize",
            "input": args,
            "count": response["response"]["claims"],
            "ms": trace[0].get("ms") or 0,
        }]
        return {**result, "calls_made": calls_made, "pending": None, "trace": trace}

    return {
        "calls_made": calls_made,
        "pending": None,
        "trace": [{
            "node": "agent",
            "title": "Caller",
            "message": f"Unknown tool {name}.",
            "ms": int((time.perf_counter() - started) * 1000),
        }],
    }


def answer_from_loop(state: LoopState) -> dict:
    started = time.perf_counter()
    if not state.get("hits"):
        answer = _empty_answer(state, "The tool loop found no public pages to cite.")
        message = "No pages to cite."
    else:
        answer = _compose_answer(state)
        message = "Brief ready, with sources."
    return {
        "answer": answer,
        "trace": [{
            "node": "answer",
            "title": "Answer",
            "message": message,
            "ms": int((time.perf_counter() - started) * 1000),
        }],
    }


def route_after_caller(state: LoopState) -> str:
    if state.get("pending"):
        return "tools"
    return "answer"


def build_loop():
    graph = StateGraph(LoopState)
    graph.add_node("caller", caller_node)
    graph.add_node("tools", tools_node)
    graph.add_node("answer", answer_from_loop)
    graph.add_edge(START, "caller")
    graph.add_conditional_edges("caller", route_after_caller, {"tools": "tools", "answer": "answer"})
    graph.add_edge("tools", "caller")
    graph.add_edge("answer", END)
    return graph.compile()


loop = build_loop()


def stream_tool_loop(query: str):
    query = " ".join((query or "").split())
    if not query:
        yield {"type": "error", "message": "Ask a question first."}
        return
    yield {"type": "start", "query": query, "mode": "loop"}
    started = time.perf_counter()
    try:
        for update in loop.stream({"query": query, "trace": []}, stream_mode="updates"):
            node, data = next(iter(update.items()))
            trace = (data.get("trace") or [{}])[-1]
            public_node = trace.get("node") or node
            if public_node == "answer" or node == "answer":
                yield {"type": "answer", "node": "answer", "trace": trace, "answer": data.get("answer")}
            else:
                yield {"type": "step", "node": public_node, "trace": trace}
        yield {"type": "done", "ms": int((time.perf_counter() - started) * 1000), "mode": "loop"}
    except Exception as exc:
        yield {"type": "error", "message": f"{type(exc).__name__}: {exc}"}


def _ask(started: float, pending: dict, message: str) -> dict:
    return {
        "pending": pending,
        "trace": [{
            "node": "agent",
            "title": "Caller",
            "message": message,
            "function_call": pending,
            "ms": int((time.perf_counter() - started) * 1000),
        }],
    }


def _stop(started: float, message: str) -> dict:
    return {
        "pending": None,
        "trace": [{
            "node": "agent",
            "title": "Caller",
            "message": message,
            "ms": int((time.perf_counter() - started) * 1000),
        }],
    }


def _run_search(state: LoopState, args: dict, call_id: str, calls_made: int, started: float) -> dict:
    query = args.get("query") or state["query"]
    try:
        results = search_web(query, max_results=int(args.get("max_results") or 5))
        error = ""
    except Exception as exc:
        results = []
        error = f"{type(exc).__name__}: {exc}"
    merged = {hit["url"]: hit for hit in (state.get("hits") or [])}
    for hit in results:
        merged.setdefault(hit["url"], hit)
    hits = list(merged.values())[:6]
    response = function_response("search_web", call_id, {"count": len(results), "error": error})
    return {
        "hits": hits,
        "query_index": (state.get("query_index") or 0) + 1,
        "calls_made": calls_made,
        "pending": None,
        "trace": [{
            "node": "search",
            "title": "Search",
            "tool": "search_web",
            "message": f"search_web returned {len(results)} pages.",
            "function_response": response,
            "calls": [{
                "tool": "search_web",
                "input": {"query": query, "max_results": args.get("max_results") or 5},
                "count": len(results),
                "error": error,
                "ms": int((time.perf_counter() - started) * 1000),
                "results": [
                    {"title": hit.get("title"), "url": hit.get("url"), "snippet": (hit.get("snippet") or "")[:160]}
                    for hit in results[:4]
                ],
            }],
            "ms": int((time.perf_counter() - started) * 1000),
        }],
    }


def _run_fetch(state: LoopState, args: dict, call_id: str, calls_made: int, started: float) -> dict:
    url = args.get("url") or ""
    hit = next((item for item in (state.get("hits") or []) if item.get("url") == url), {"url": url, "title": "", "snippet": ""})
    try:
        fetched = fetch_page(url)
    except Exception as exc:
        fetched = {"url": url, "ok": False, "error": str(exc), "text": "", "words": 0, "ms": 0, "title": "", "domain": ""}
    page = _page_from_hit(hit, fetched)
    pages = list(state.get("pages") or [])
    page["id"] = len(pages) + 1
    pages.append(page)
    response = function_response("fetch_page", call_id, {"ok": page["ok"], "words": page["words"], "title": page["title"]})
    return {
        "pages": pages,
        "fetch_index": (state.get("fetch_index") or 0) + 1,
        "calls_made": calls_made,
        "pending": None,
        "trace": [{
            "node": "read",
            "title": "Read",
            "tool": "fetch_page",
            "message": f"Read {page['title'] or url}." if page["ok"] else f"Could not read {url}.",
            "function_response": response,
            "calls": [{
                "tool": "fetch_page",
                "input": {"url": url},
                "ok": page["ok"],
                "title": page["title"],
                "words": page["words"],
                "error": page["error"],
                "ms": page["ms"],
            }],
            "ms": int((time.perf_counter() - started) * 1000),
        }],
    }
