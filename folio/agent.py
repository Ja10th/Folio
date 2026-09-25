"""Folio's graph.

User
 ↓
Agent          interpret the question, write 2 searches
 ↓
Search         search_web()
 ↓
Read pages     fetch_page()
 ↓
Summarize      summarize()
 ↓
Answer with sources

One LangGraph. Three tools. The path is fixed on purpose.
"""

from __future__ import annotations

import operator
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Annotated, Any, TypedDict
from zoneinfo import ZoneInfo

from langgraph.graph import END, START, StateGraph

from folio.tools import (
    diversify,
    domain_of,
    fetch_page,
    is_public_http,
    normalize_url,
    rank_hit,
    search_web,
    summarize,
    is_official_url,
)

KICKER = {
    "define": "What it is",
    "how": "How it works",
    "compare": "Side by side",
    "general": "Research brief",
}
POINTS_LABEL = {
    "define": "What it does",
    "how": "What the pages say",
    "compare": "Where they differ",
    "general": "What the pages say",
}


class ResearchState(TypedDict, total=False):
    query: str
    topic: str
    intent: str
    rationale: str
    search_queries: list[str]
    hits: list[dict]
    pages: list[dict]
    summary: dict
    answer: dict
    trace: Annotated[list[dict], operator.add]


def interpret(query: str) -> dict[str, Any]:
    """Turn a request into a topic, an intent, and two search queries."""
    text = " ".join(query.strip().split())
    lower = text.lower()

    if any(token in lower for token in (" vs ", " versus ", "difference between", "compared to", "compare ")):
        intent = "compare"
    elif lower.startswith("how ") or " how " in f" {lower} ":
        intent = "how"
    elif any(token in lower for token in ("what is", "what are", "what does", "what do", "what's", "who is", "who are", "tell me what", "what it does")):
        intent = "define"
    else:
        intent = "general"

    topic = re.sub(
        r"^(?:please\s+)?(?:can\s+you\s+|could\s+you\s+)?"
        r"(?:research|look\s+up|investigate|find\s+out(?:\s+about)?|"
        r"explain|describe|summarize|summarise)\s+",
        "",
        text,
        flags=re.I,
    ).strip()
    topic = re.sub(
        r"\s*,?\s+(?:and\s+)?(?:then\s+)?(?:tell\s+me|explain|describe|show\s+me|summarize|summarise|give\s+me)\b.*$",
        "",
        topic,
        flags=re.I,
    ).strip()
    topic = re.sub(r",?\s+in\s+plain\s+(?:language|english)\s*$", "", topic, flags=re.I).strip(" ?.,")
    topic = re.sub(r"^(?:what|who)\s+(?:is|are|'s)\s+", "", topic, flags=re.I).strip(" ?.,")
    topic = re.sub(r"^how\s+(?:does|do|did)\s+", "", topic, flags=re.I).strip(" ?.,")
    topic = re.sub(r"^what\s+does\s+", "", topic, flags=re.I).strip()
    topic = re.sub(r"\s+do\??(?:\s+.*)?$", "", topic, flags=re.I).strip(" ?.,")
    topic = re.sub(r"^go deeper on\s+", "", topic, flags=re.I).strip()
    topic = re.sub(r"\.\s+what does it actually do\b.*$", "", topic, flags=re.I).strip(" ?.,")
    topic = re.sub(r"^the\s+", "", topic, flags=re.I).strip()
    if len(topic) < 2:
        topic = text.rstrip("?.")

    if intent == "define":
        queries = [f"what is {topic}", f"{topic} official documentation"]
    elif intent == "how":
        subject = re.sub(
            r"\s+(handle|handles|work|works|use|uses|do|does|call|calls|manage|manages)\b.*$",
            "",
            topic,
            flags=re.I,
        ).strip(" ?.,") or topic
        extra = f"{subject} tools" if "tool" in text.lower() else f"{subject} explained"
        queries = [text.rstrip("?."), extra]
        topic = subject
    elif intent == "compare":
        queries = [text.rstrip("?."), f"{topic} comparison"]
    else:
        queries = [text.rstrip("?."), f"{topic} overview"]

    deduped = []
    seen = set()
    for item in queries:
        key = item.lower().strip()
        if len(key) < 3 or key in seen:
            continue
        seen.add(key)
        deduped.append(item.strip())
    if len(deduped) == 1:
        deduped.append(f"{topic} overview")
    return {"topic": topic[:140], "intent": intent, "queries": deduped[:2]}


def agent_node(state: ResearchState) -> dict:
    started = time.perf_counter()
    plan = interpret(state["query"])
    label = {
        "define": "a definitional brief",
        "how": "a how-it-works brief",
        "compare": "a comparison",
        "general": "a short research brief",
    }[plan["intent"]]
    rationale = (
        f"This is {label} on {plan['topic']}. "
        "I'll search, read the strongest pages, and summarize only what they say."
    )
    return {
        "topic": plan["topic"],
        "intent": plan["intent"],
        "search_queries": plan["queries"],
        "rationale": rationale,
        "trace": [{
            "node": "agent",
            "title": "Agent",
            "message": rationale,
            "topic": plan["topic"],
            "intent": plan["intent"],
            "queries": plan["queries"],
            "ms": int((time.perf_counter() - started) * 1000),
        }],
    }


def search_node(state: ResearchState) -> dict:
    started = time.perf_counter()
    queries = state.get("search_queries") or [state["query"]]
    topic = state.get("topic") or state["query"]
    topic_terms = {
        word for word in re.findall(r"[a-z0-9][a-z0-9.+-]{2,}", topic.lower())
        if word not in {"the", "and", "for", "how", "what", "with"}
    }
    calls = []
    merged: dict[str, dict] = {}
    for query in queries:
        t0 = time.perf_counter()
        error = ""
        try:
            results = search_web(query, max_results=5)
        except Exception as exc:
            results = []
            error = f"{type(exc).__name__}: {exc}"
        calls.append({
            "tool": "search_web",
            "input": {"query": query, "max_results": 5},
            "count": len(results),
            "error": error,
            "ms": int((time.perf_counter() - t0) * 1000),
            "results": [
                {"title": hit["title"], "url": hit["url"], "snippet": hit["snippet"][:200]}
                for hit in results[:5]
            ],
        })
        for index, hit in enumerate(results):
            if not is_public_http(hit["url"]):
                continue
            key = normalize_url(hit["url"])
            score = rank_hit(hit, topic, topic_terms, index)
            if key in merged:
                merged[key]["score"] += 1.3
                if len(hit["snippet"]) > len(merged[key]["snippet"]):
                    merged[key]["snippet"] = hit["snippet"]
                continue
            merged[key] = {
                **hit,
                "domain": domain_of(hit["url"]),
                "score": score,
            }

    if not any(is_official_url(hit["url"]) for hit in merged.values()):
        extra = f"{topic} official documentation"
        if extra.lower() not in {q.lower() for q in queries}:
            queries = [*queries, extra]
            t0 = time.perf_counter()
            error = ""
            try:
                results = search_web(extra, max_results=5)
            except Exception as exc:
                results = []
                error = f"{type(exc).__name__}: {exc}"
            calls.append({
                "tool": "search_web",
                "input": {"query": extra, "max_results": 5},
                "count": len(results),
                "error": error,
                "ms": int((time.perf_counter() - t0) * 1000),
                "results": [
                    {"title": hit["title"], "url": hit["url"], "snippet": hit["snippet"][:200]}
                    for hit in results[:5]
                ],
            })
            for index, hit in enumerate(results):
                if not is_public_http(hit["url"]):
                    continue
                key = normalize_url(hit["url"])
                score = rank_hit(hit, topic, topic_terms, index) + 1.5
                if key in merged:
                    merged[key]["score"] += 1.3
                    continue
                merged[key] = {**hit, "domain": domain_of(hit["url"]), "score": score}

    ranked = sorted(merged.values(), key=lambda hit: hit["score"], reverse=True)
    hits = diversify(ranked, limit=6)
    if hits:
        message = f"{len(hits)} distinct pages from {len(calls)} searches."
    else:
        message = "search_web returned nothing usable."
    return {
        "hits": hits,
        "trace": [{
            "node": "search",
            "title": "Search",
            "tool": "search_web",
            "message": message,
            "calls": calls,
            "hits": [_public_hit(hit) for hit in hits],
            "ms": int((time.perf_counter() - started) * 1000),
        }],
    }


def route_after_search(state: ResearchState) -> str:
    if state.get("hits"):
        return "read"
    return "answer"


def _reading_list(hits: list[dict], limit: int = 4) -> list[dict]:
    """Read a few pages, and don't skip the docs page if search found one."""
    if not hits:
        return []
    official = next((hit for hit in hits if is_official_url(hit["url"])), None)
    ordered = [official] if official else []
    for hit in hits:
        if official and hit["url"] == official["url"]:
            continue
        ordered.append(hit)
        if len(ordered) >= limit:
            break
    return ordered[:limit]


def _fetch_hits(hits: list[dict]) -> list[dict]:
    if not hits:
        return []
    pages: list[dict | None] = [None] * len(hits)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(fetch_page, hit["url"]): index for index, hit in enumerate(hits)}
        for future in as_completed(futures):
            index = futures[future]
            hit = hits[index]
            try:
                fetched = future.result()
            except Exception as exc:
                fetched = {
                    "url": hit["url"],
                    "title": "",
                    "domain": hit.get("domain") or "",
                    "text": "",
                    "words": 0,
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "ms": 0,
                }
            pages[index] = _page_from_hit(hit, fetched)
    return [page for page in pages if page]


def read_node(state: ResearchState) -> dict:
    started = time.perf_counter()
    hits = list(state.get("hits") or [])
    targets = _reading_list(hits, limit=4)
    pages = _fetch_hits(targets)

    for index, page in enumerate(pages, start=1):
        page["id"] = index

    ok = sum(1 for page in pages if page["ok"])
    message = f"Read {ok} of {len(pages)} pages."
    if ok < len(pages):
        message += " Unreadable ones keep their search snippet."
    return {
        "pages": pages,
        "trace": [{
            "node": "read",
            "title": "Read",
            "tool": "fetch_page",
            "message": message,
            "calls": [
                {
                    "tool": "fetch_page",
                    "input": {"url": page["url"]},
                    "ok": page["ok"],
                    "title": page["title"],
                    "words": page["words"],
                    "error": page["error"],
                    "ms": page["ms"],
                }
                for page in pages
            ],
            "ms": int((time.perf_counter() - started) * 1000),
        }],
    }


def summarize_node(state: ResearchState) -> dict:
    started = time.perf_counter()
    pages = state.get("pages") or []
    documents = []
    for page in pages:
        text = page["text"] if page.get("ok") else (page.get("snippet") or page.get("text") or "")
        documents.append({
            "id": page["id"],
            "title": page.get("title") or page.get("domain") or "Untitled",
            "url": page["url"],
            "domain": page.get("domain") or "",
            "text": text,
            "ok": bool(page.get("ok")),
        })
    summary = summarize(state["query"], documents, intent=state.get("intent") or "general")
    words = sum(page.get("words") or 0 for page in pages)
    claims = summary.get("claims") or 0
    return {
        "summary": summary,
        "trace": [{
            "node": "summarize",
            "title": "Summarize",
            "tool": "summarize",
            "message": f"Kept {claims} claims from {words:,} words.",
            "calls": [{
                "tool": "summarize",
                "input": {"documents": len(documents), "query": state["query"][:160]},
                "count": claims,
                "ms": int((time.perf_counter() - started) * 1000),
            }],
            "ms": int((time.perf_counter() - started) * 1000),
        }],
    }


def answer_node(state: ResearchState) -> dict:
    started = time.perf_counter()
    if not state.get("hits"):
        answer = _empty_answer(
            state,
            "I didn't find public pages for that. Try a more specific name, or the full question.",
        )
        message = "No pages to cite."
    else:
        answer = _compose_answer(state)
        message = "Brief ready, with sources." if not answer.get("failed") else "Thin sources."
    return {
        "answer": answer,
        "trace": [{
            "node": "answer",
            "title": "Answer",
            "message": message,
            "ms": int((time.perf_counter() - started) * 1000),
        }],
    }


def build_graph():
    graph = StateGraph(ResearchState)
    graph.add_node("agent", agent_node)
    graph.add_node("search", search_node)
    graph.add_node("read", read_node)
    graph.add_node("summarize", summarize_node)
    graph.add_node("answer", answer_node)

    graph.add_edge(START, "agent")
    graph.add_edge("agent", "search")
    graph.add_conditional_edges(
        "search",
        route_after_search,
        {"read": "read", "answer": "answer"},
    )
    graph.add_edge("read", "summarize")
    graph.add_edge("summarize", "answer")
    graph.add_edge("answer", END)
    return graph.compile()


graph = build_graph()


def stream_research(query: str):
    """Yield UI events as each node finishes."""
    query = " ".join((query or "").split())
    if not query:
        yield {"type": "error", "message": "Ask a question first."}
        return
    if len(query) > 500:
        yield {"type": "error", "message": "Keep the question under 500 characters."}
        return

    yield {"type": "start", "query": query}
    started = time.perf_counter()
    try:
        for update in graph.stream({"query": query, "trace": []}, stream_mode="updates"):
            node, data = next(iter(update.items()))
            trace = (data.get("trace") or [{}])[-1]
            if node == "answer":
                yield {"type": "answer", "node": node, "trace": trace, "answer": data.get("answer")}
            else:
                yield {"type": "step", "node": node, "trace": trace}
        yield {"type": "done", "ms": int((time.perf_counter() - started) * 1000)}
    except Exception as exc:
        yield {"type": "error", "message": f"{type(exc).__name__}: {exc}"}


def mermaid() -> str:
    return graph.get_graph().draw_mermaid()


# --- answer composition ---------------------------------------------------

def _compose_answer(state: ResearchState) -> dict:
    pages = state.get("pages") or []
    summary = state.get("summary") or {}
    sources = [_public_source(page) for page in pages]
    cited = set()

    lead = summary.get("lead")
    points = list(summary.get("points") or [])
    quote = summary.get("quote")

    if not lead:
        # Last resort: the search snippets, still cited.
        points = []
        for source in sources:
            snippet = (source.get("snippet") or "").strip()
            if snippet:
                points.append({"text": snippet if snippet.endswith(".") else snippet + ".", "sources": [source["id"]]})
        if points:
            lead = points.pop(0)
        else:
            return _empty_answer(state, "The pages opened, but none of them yielded a sentence I could cite.")

    for claim in [lead, *points, *([quote] if quote else [])]:
        cited.update(claim.get("sources") or [])
    for source in sources:
        source["cited"] = source["id"] in cited

    note = _note(state, pages)
    answer = {
        "failed": False,
        "query": state.get("query") or "",
        "topic": state.get("topic") or state.get("query"),
        "intent": state.get("intent") or "general",
        "kicker": KICKER.get(state.get("intent") or "", "Research brief"),
        "points_label": POINTS_LABEL.get(state.get("intent") or "", "What the pages say"),
        "queries": state.get("search_queries") or [],
        "lead": lead,
        "points": points,
        "quote": quote,
        "sources": sources,
        "note": note,
        "date": _today(),
        "stats": {
            "pages": len(pages),
            "words": sum(page.get("words") or 0 for page in pages),
            "claims": 1 + len(points) + (1 if quote else 0),
            "read": sum(1 for page in pages if page.get("ok")),
        },
    }
    answer["plain"] = _plain(answer)
    return answer


def _empty_answer(state: ResearchState, message: str) -> dict:
    answer = {
        "failed": True,
        "topic": state.get("topic") or "No results",
        "intent": state.get("intent") or "general",
        "kicker": "Nothing to cite",
        "queries": state.get("search_queries") or [],
        "lead": {"text": message, "sources": []},
        "points": [],
        "quote": None,
        "sources": [],
        "note": "search_web did not return a usable public page.",
        "date": _today(),
        "stats": {"pages": 0, "words": 0, "claims": 0, "read": 0},
    }
    answer["plain"] = message
    return answer


def _note(state: ResearchState, pages: list[dict]) -> str:
    queries = state.get("search_queries") or []
    quoted = " and ".join(f"“{q}”" for q in queries) if queries else "the question"
    read = sum(1 for page in pages if page.get("ok"))
    missed = len(pages) - read
    sentence = f"Searched {quoted}. Read {read} of {len(pages)} pages."
    if missed:
        sentence += " Where a page would not open, only its search snippet was used."
    sentence += " Every claim is a sentence from those pages, not something added from memory."
    return sentence


def _plain(answer: dict) -> str:
    lines = [
        answer["topic"],
        answer["kicker"],
        "",
        _cite_line(answer["lead"]["text"], answer["lead"].get("sources")),
        "",
    ]
    for point in answer["points"]:
        lines.append("• " + _cite_line(point["text"], point.get("sources")))
    if answer.get("quote"):
        lines.append("")
        lines.append(_cite_line("“" + answer["quote"]["text"] + "”", answer["quote"].get("sources")))
    lines.append("")
    lines.append("Sources")
    for source in answer["sources"]:
        lines.append(f"[{source['id']}] {source['title']}")
        lines.append(f"    {source['url']}")
    lines.append("")
    lines.append(answer["note"])
    return "\n".join(lines)


def _cite_line(text: str, sources: list[int] | None) -> str:
    if not sources:
        return text
    marks = "".join(f"[{n}]" for n in sources)
    return f"{text} {marks}"


def _public_hit(hit: dict) -> dict:
    return {
        "title": hit.get("title") or hit.get("url"),
        "url": hit["url"],
        "domain": hit.get("domain") or domain_of(hit["url"]),
        "snippet": (hit.get("snippet") or "")[:240],
    }


def _public_source(page: dict) -> dict:
    snippet = (page.get("snippet") or "").strip()
    if not snippet:
        snippet = " ".join((page.get("text") or "").split())[:240]
    return {
        "id": page["id"],
        "title": page.get("title") or page.get("domain") or "Untitled",
        "url": page["url"],
        "domain": page.get("domain") or domain_of(page["url"]),
        "snippet": snippet[:280],
        "words": page.get("words") or 0,
        "ok": bool(page.get("ok")),
        "error": page.get("error") or "",
    }


def _page_from_hit(hit: dict, fetched: dict) -> dict:
    title = fetched.get("title") or hit.get("title") or fetched.get("domain") or "Untitled"
    text = fetched.get("text") or ""
    if not fetched.get("ok"):
        text = hit.get("snippet") or text
    return {
        "url": fetched.get("url") or hit["url"],
        "title": title,
        "domain": fetched.get("domain") or hit.get("domain") or domain_of(hit["url"]),
        "snippet": hit.get("snippet") or "",
        "text": text,
        "words": fetched.get("words") or 0,
        "ok": bool(fetched.get("ok")),
        "error": fetched.get("error") or "",
        "ms": fetched.get("ms") or 0,
    }


def _today() -> str:
    try:
        now = datetime.now(ZoneInfo("Africa/Lagos"))
    except Exception:
        now = datetime.now()
    return now.strftime("%d %b %Y")
