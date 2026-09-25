"""The three tools.

search_web()   find pages
fetch_page()   read one page
summarize()    condense those pages into sourced claims

Nothing here calls a model. summarize() only keeps sentences that
were actually on the page.
"""

from __future__ import annotations

import html
import re
import time
from collections import defaultdict
from typing import Any
from urllib.parse import parse_qsl, unquote, urlencode, urlparse, urlunparse

import httpx
import trafilatura
from bs4 import BeautifulSoup

try:
    from ddgs import DDGS
except ImportError:  # older package name
    from duckduckgo_search import DDGS  # type: ignore


UA = (
    "Mozilla/5.0 (compatible; FolioResearch/1.0) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)
HEADERS = {
    "User-Agent": UA,
    "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
TIMEOUT = httpx.Timeout(12.0, connect=5.0)
MAX_PAGE_CHARS = 12_000
SKIP_EXTENSIONS = (
    ".pdf", ".zip", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg",
    ".mp4", ".mp3", ".wav", ".docx", ".pptx", ".xlsx",
)
DOWNRANK_HOSTS = (
    "pinterest.", "facebook.", "instagram.", "tiktok.", "reddit.com",
    "quora.com", "youtube.com", "youtu.be", "linkedin.com", "twitter.com",
    "x.com", "slideshare.net", "scribd.com",
)
STOPWORDS = frozenset(
    """
    a an the and or but if then else when while of to for from in on at by
    with without into over after before about as is are was were be been being
    it its this that these those they them their we you your i me my our not
    no nor so than too very can could should would will just also more most
    other such only own same do does did doing done have has had having what
    which who whom whose why how where when research tell explain describe
    please page home using used use make makes made get gets
    """.split()
)
JUNK = re.compile(
    r"\b(cookie|subscribe|sign up|newsletter|all rights reserved|privacy policy|"
    r"terms of use|share this|related articles|advertisement|table of contents|"
    r"skip to content|was this helpful|follow us|accept all|log in|sign in|"
    r"click here|read more|learn more|get started|book a demo)\b",
    re.I,
)
FLUFF = re.compile(
    r"\b(analogy|metaphor|imagine|let's|lets|welcome|bonus|if you like|"
    r"in this (?:article|post|guide|section|piece)|this analogy|"
    r"as you can see|excited to|cartographer)\b",
    re.I,
)
VAGUE = re.compile(
    r"\b(at its core|the power of|intricate|versatile platform|these technologies|"
    r"this approach|rich, personalized|leading open[- ]source|unblocks?|"
    r"cutting-edge|seamless|next-generation|game-changing|world-class|"
    r"in today's|revolutioni[sz]e|illuminat\w*|empower\w*|unlock\w*|"
    r"sacrific\w*|without slowing|full transparency|scale artificial|"
    r"seamlessly|production-ready|confidently)\b",
    re.I,
)
MECHANISM = re.compile(
    r"\b(nodes?|edges?|state|states|graph|graphs|workflow|workflows|"
    r"checkpoint\w*|persist\w*|stream\w*|human-in-the-loop|tool calls?|"
    r"durable|orchestrat\w*|runtime|memory|deterministic|agentic|oversight)\b",
    re.I,
)
CONCRETE = re.compile(
    r"\b(provides|uses|stores|enables|supports|combines|allows|manages|"
    r"defines|persists|orchestrat\w+|designed|built|builds|includes|"
    r"connects|routes|executes|maintains|handles)\b",
    re.I,
)
ABBREV = re.compile(r"\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|e\.g|i\.e|U\.S|U\.K)\.")


def search_web(query: str, max_results: int = 5) -> list[dict[str, str]]:
    """Search the public web. Each hit has title, url, and snippet."""
    query = " ".join((query or "").split())
    if not query:
        return []
    results = _search_ddgs(query, max_results)
    if not results:
        results = _search_ddg_html(query, max_results)
    clean = []
    seen = set()
    for hit in results:
        url = (hit.get("url") or "").strip()
        if not url or not is_public_http(url) or _skip_url(url):
            continue
        key = normalize_url(url)
        if key in seen:
            continue
        seen.add(key)
        title = _one_line(hit.get("title") or url)
        snippet = _one_line(hit.get("snippet") or "")
        clean.append({"title": title[:180], "url": url, "snippet": snippet[:320]})
        if len(clean) >= max_results:
            break
    return clean


def fetch_page(url: str) -> dict[str, Any]:
    """Download a public page and extract its readable text."""
    started = time.perf_counter()
    base = {
        "url": url,
        "title": "",
        "domain": domain_of(url),
        "text": "",
        "words": 0,
        "ok": False,
        "error": "",
        "ms": 0,
    }
    if not is_public_http(url) or _skip_url(url):
        base["error"] = "URL is not a public article."
        base["ms"] = int((time.perf_counter() - started) * 1000)
        return base
    try:
        with httpx.Client(timeout=TIMEOUT, follow_redirects=True, headers=HEADERS) as client:
            response = client.get(url)
            response.raise_for_status()
            final = str(response.url)
            if not is_public_http(final):
                base["error"] = "Redirected to a non-public address."
                base["ms"] = int((time.perf_counter() - started) * 1000)
                return base
            ctype = (response.headers.get("content-type") or "").lower()
            if "pdf" in ctype or "image/" in ctype or "audio/" in ctype or "video/" in ctype:
                base["error"] = "Not an HTML article."
                base["ms"] = int((time.perf_counter() - started) * 1000)
                return base
            raw = response.text[:1_500_000]
    except Exception as exc:  # network, HTTP, decode
        base["error"] = f"{type(exc).__name__}: {exc}"
        base["ms"] = int((time.perf_counter() - started) * 1000)
        return base

    title = _html_title(raw)
    text = _extract_text(raw, final)
    text = _clip(text, MAX_PAGE_CHARS)
    words = len(text.split())
    base.update({
        "url": final,
        "title": title,
        "domain": domain_of(final),
        "text": text,
        "words": words,
        "ok": words >= 40,
        "error": "" if words >= 40 else "Page had almost no readable text.",
        "ms": int((time.perf_counter() - started) * 1000),
    })
    return base


def summarize(query: str, documents: list[dict], intent: str = "general") -> dict:
    """Condense fetched pages into sourced claims.

    Every claim is a sentence from a document. Sources are 1-based document ids.
    """
    topic = _topic_phrase(query)
    query_terms = _content_words(query) | _content_words(topic)
    candidates = []
    for doc in documents:
        text = (doc.get("text") or "").strip()
        if not text:
            continue
        weight = 1.0 if doc.get("ok", True) else 0.62
        official = _officialish(doc.get("url") or "")
        if official:
            weight += 0.22
        about = bool(topic) and topic.lower() in " ".join(
            (doc.get("title") or "", doc.get("url") or "", doc.get("domain") or "")
        ).lower()
        for index, sentence in enumerate(_sentences(text)):
            scored = _score_sentence(
                sentence, query_terms, topic, intent, index, weight, about=about,
            )
            if scored is None:
                continue
            score, words = scored
            candidates.append({
                "text": sentence,
                "sources": [int(doc["id"])],
                "score": score,
                "words": words,
                "domain": doc.get("domain") or "",
                "official": official,
            })

    clusters = _cluster(candidates)
    if not clusters:
        return {"lead": None, "points": [], "quote": None, "claims": 0}

    lead = _pick_lead(clusters, topic, intent)
    selected = [lead]
    points = []
    pool = [c for c in clusters if c is not lead]
    domain_counts: dict[str, int] = defaultdict(int)
    while len(points) < 4 and pool:
        nxt = _mmr_next(pool, selected)
        if nxt is None:
            break
        # Keep a low bar so thin pages still produce a brief, but refuse scraps.
        if any(_jaccard(nxt["words"], done["words"]) >= 0.46 for done in selected):
            pool.remove(nxt)
            continue
        domain = nxt.get("domain") or ""
        cap = 3 if nxt.get("official") else 2
        if domain and domain_counts[domain] >= cap and len(points) >= 2:
            pool.remove(nxt)
            continue
        if nxt["score"] < 0.85 and len(points) >= 2:
            break
        if nxt["score"] < 0.45:
            break
        points.append({"text": nxt["text"], "sources": nxt["sources"]})
        selected.append(nxt)
        domain_counts[domain] += 1
        pool.remove(nxt)

    quote = None
    lead_sources = list(lead["sources"])
    for other in clusters:
        if other is lead:
            continue
        if _jaccard(lead["words"], other["words"]) >= 0.45 and topic.lower() in other["text"].lower():
            for src in other["sources"]:
                if src not in lead_sources:
                    lead_sources.append(src)
            break

    return {
        "lead": {"text": lead["text"], "sources": lead_sources},
        "points": points,
        "quote": quote,
        "claims": 1 + len(points) + (1 if quote else 0),
    }


# --- search helpers -------------------------------------------------------

def _search_ddgs(query: str, max_results: int) -> list[dict[str, str]]:
    try:
        with DDGS(timeout=20) as ddgs:
            raw = list(ddgs.text(query, max_results=max_results))
    except Exception:
        return []
    hits = []
    for item in raw:
        url = item.get("href") or item.get("url") or ""
        hits.append({
            "title": item.get("title") or "",
            "url": url,
            "snippet": item.get("body") or item.get("snippet") or "",
        })
    return hits


def _search_ddg_html(query: str, max_results: int) -> list[dict[str, str]]:
    try:
        with httpx.Client(timeout=TIMEOUT, follow_redirects=True, headers=HEADERS) as client:
            response = client.post(
                "https://html.duckduckgo.com/html/",
                data={"q": query},
            )
            response.raise_for_status()
    except Exception:
        return []
    soup = BeautifulSoup(response.text, "lxml")
    hits = []
    for anchor in soup.select("a.result__a"):
        href = _unwrap_ddg(anchor.get("href") or "")
        title = anchor.get_text(" ", strip=True)
        snippet = ""
        parent = anchor.find_parent("div", class_="result") or anchor.find_parent("tr")
        if parent:
            snip = parent.select_one(".result__snippet")
            if snip:
                snippet = snip.get_text(" ", strip=True)
        if href and title:
            hits.append({"title": title, "url": href, "snippet": snippet})
        if len(hits) >= max_results:
            break
    return hits


def _unwrap_ddg(href: str) -> str:
    if href.startswith("//"):
        href = "https:" + href
    parsed = urlparse(href)
    qs = dict(parse_qsl(parsed.query))
    if "uddg" in qs:
        return unquote(qs["uddg"])
    return href


# --- fetch helpers --------------------------------------------------------

def _extract_text(html_text: str, url: str) -> str:
    extracted = trafilatura.extract(
        html_text,
        url=url,
        include_comments=False,
        include_tables=False,
        include_links=False,
        favor_precision=True,
        deduplicate=True,
    ) or ""
    if len(extracted) >= 400:
        return _clean_block(extracted)
    soup = BeautifulSoup(html_text, "lxml")
    for tag in soup(["script", "style", "noscript", "svg", "nav", "footer", "form", "aside"]):
        tag.decompose()
    node = soup.find("article") or soup.find("main") or soup.body
    rough = node.get_text("\n", strip=True) if node else ""
    rough = _clean_block(rough)
    if len(rough) > len(extracted):
        return rough
    return _clean_block(extracted)


def _html_title(html_text: str) -> str:
    soup = BeautifulSoup(html_text, "lxml")
    if soup.title and soup.title.string:
        return _one_line(soup.title.string)[:180]
    return ""


def _clean_block(text: str) -> str:
    text = html.unescape(text or "")
    text = text.replace("\u00a0", " ").replace("\u200b", "")
    lines = []
    for line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", line).strip()
        if not line or JUNK.search(line):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


# --- summarize helpers ----------------------------------------------------

def _sentences(text: str) -> list[str]:
    found = []
    for para in re.split(r"\n+", text):
        para = para.strip()
        if not para:
            continue
        para = re.sub(r"^(?:\d+[\).\]]\s+|[-•–—]\s+)", "", para)
        protected = ABBREV.sub(lambda m: m.group(0).replace(".", "∯"), para)
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9“\"'])", protected)
        for part in parts:
            sentence = part.replace("∯", ".").strip()
            sentence = sentence.strip(" \t\"“”")
            sentence = re.sub(r"\s+", " ", sentence)
            if sentence and sentence[-1] not in ".!?":
                if sentence.endswith(":"):
                    continue
                sentence += "."
            sentence = re.sub(r"^([A-Za-z][A-Za-z-]{2,24}):\s+\1\b", r"\1", sentence)
            if sentence:
                found.append(sentence)
    return found


def _score_sentence(sentence, query_terms, topic, intent, index, weight, about=False):
    if not (50 <= len(sentence) <= 360):
        return None
    if JUNK.search(sentence):
        return None
    letters = sum(ch.isalpha() for ch in sentence)
    if letters / max(len(sentence), 1) < 0.62:
        return None
    if max((len(word) for word in sentence.split()), default=0) > 40:
        return None
    if sentence.count("/") >= 2 and sentence.count(" ") < 10:
        return None
    if sentence.endswith("?"):
        return None
    if FLUFF.search(sentence) or VAGUE.search(sentence):
        return None
    if re.match(r"^(in this|this (?:article|post|guide|section|analogy|piece)|today we|here we)\b", sentence, re.I):
        return None
    if re.match(r"^(Design|Discover|Explore|Learn|See how|Get started|Try )\b", sentence):
        return None
    if re.search(r"\b(for more information|see also|see the documentation|in the .+ documentation)\b", sentence, re.I):
        return None
    if re.search(r"\bare a function\b|\bwithin Python\b", sentence, re.I):
        score_penalty = 1.6
    else:
        score_penalty = 0.0
    if re.search(r"\b(import|def|return|class)\b|[=]{2}|\{|\}|\*\*", sentence):
        return None
    words = _content_words(sentence)
    if len(words) < 6:
        return None
    overlap = len(words & query_terms)
    coverage = overlap / max(len(query_terms), 1)
    # Repeating the name should help, not drown out a docs page that states the mechanism once.
    score = min(2.0, coverage * 2.2)
    named = bool(topic) and topic.lower() in sentence.lower()
    if named:
        score += 0.75
    elif about:
        score += 0.85
    if intent == "define" and re.search(r"\b(is|are)\s+(an?|the)\b", sentence, re.I):
        score += 1.3
    if intent == "how" and re.search(r"\b(by|through|via|using|works|step)\b", sentence, re.I):
        score += 0.45
    if intent == "how" and re.search(r"\b(tools?|handoffs?|function calls?)\b", sentence, re.I):
        if query_terms & {"tool", "tools", "handoff", "handoffs", "function"}:
            score += 1.7
    if index < 2:
        score += 0.55
    elif index < 5:
        score += 0.25
    if 80 <= len(sentence) <= 230:
        score += 0.45
    if re.match(r"^(It|This|That|These|Those|They|There)\b", sentence):
        score -= 0.35
    if re.match(r"^(Design|Build|Create|Discover|Explore|Learn|See|Get|Make|Try|Start)\b", sentence):
        score -= 1.4
    if CONCRETE.search(sentence):
        score += 0.65
    mechanisms = len(MECHANISM.findall(sentence))
    if mechanisms:
        score += min(1.15, 0.38 * mechanisms)
    you_count = len(re.findall(r"\b(you|your|you're)\b", sentence, re.I))
    if you_count and intent != "how":
        score -= 0.8 * you_count
    if re.search(r"https?://|www\.", sentence):
        score -= 1.4
    score -= score_penalty
    score *= weight
    if score < 0.35 and overlap == 0:
        return None
    return score, words


def _cluster(candidates: list[dict]) -> list[dict]:
    clusters = []
    for cand in sorted(candidates, key=lambda c: c["score"], reverse=True):
        placed = False
        for cluster in clusters:
            if _same_claim(cand, cluster):
                for src in cand["sources"]:
                    if src not in cluster["sources"]:
                        cluster["sources"].append(src)
                placed = True
                break
        if not placed:
            clusters.append({
                "text": cand["text"],
                "sources": list(cand["sources"]),
                "score": cand["score"],
                "words": cand["words"],
                "domain": cand["domain"],
                "official": cand.get("official", False),
            })
    return clusters


def _same_claim(a: dict, b: dict) -> bool:
    if _jaccard(a["words"], b["words"]) >= 0.52:
        return True
    # Two "X is a …" sentences about the same thing are one claim, even if worded differently.
    if _is_definition(a["text"]) and _is_definition(b["text"]) and len(a["words"] & b["words"]) >= 5:
        return True
    return False


def _is_definition(text: str) -> bool:
    return bool(re.search(r"\b(is|are)\s+(an?|the)\b", text, re.I))


def _pick_lead(clusters: list[dict], topic: str, intent: str) -> dict:
    def lead_key(cluster: dict) -> float:
        bonus = 0.0
        text = cluster["text"]
        if topic and topic.lower() in text.lower():
            bonus += 0.8
        if re.search(r"\b(is|are)\s+(an?|the)\b", text, re.I):
            bonus += 1.1 if intent == "define" else 0.4
        if re.match(r"^(It|This|That|These|Those|They)\b", text):
            bonus -= 1.0
        if not (70 <= len(text) <= 260):
            bonus -= 0.4
        return cluster["score"] + bonus

    return max(clusters, key=lead_key)


def _mmr_next(pool: list[dict], selected: list[dict]) -> dict | None:
    best = None
    best_score = -1e9
    used_domains = {c["domain"] for c in selected if c.get("domain")}
    for cand in pool:
        redundancy = max((_jaccard(cand["words"], s["words"]) for s in selected), default=0)
        diversity = 0.32 if cand.get("domain") and cand["domain"] not in used_domains else 0
        value = cand["score"] - 0.72 * redundancy + diversity
        if value > best_score:
            best_score = value
            best = cand
    return best


def _pick_quote(pool: list[dict], selected: list[dict], topic: str) -> dict | None:
    best = None
    for cand in pool:
        length = len(cand["text"])
        if not (90 <= length <= 220):
            continue
        if cand["score"] < 1.15:
            continue
        if topic and topic.lower() not in cand["text"].lower():
            continue
        if VAGUE.search(cand["text"]) or not MECHANISM.search(cand["text"]):
            continue
        if any(_jaccard(cand["words"], s["words"]) >= 0.5 for s in selected):
            continue
        if best is None or cand["score"] > best["score"]:
            best = cand
    if not best:
        return None
    return {"text": best["text"], "sources": best["sources"]}


def _content_words(text: str) -> set[str]:
    return {
        word for word in re.findall(r"[a-z0-9][a-z0-9.+-]{1,}", (text or "").lower())
        if word not in STOPWORDS and len(word) >= 3
    }


def _topic_phrase(query: str) -> str:
    text = " ".join((query or "").split())
    text = re.sub(
        r"^(?:please\s+)?(?:research|look up|investigate|explain|describe|summarize|summarise)\s+",
        "",
        text,
        flags=re.I,
    )
    text = re.split(r"\s+and\s+(?:tell|explain|describe|show)\b", text, maxsplit=1, flags=re.I)[0]
    return text.strip(" ?.")


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# --- shared url helpers ---------------------------------------------------

def is_public_http(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return False
    host = parsed.hostname.lower().rstrip(".")
    if host in {"localhost", "0.0.0.0"} or host.endswith(".local") or host.endswith(".internal"):
        return False
    try:
        import ipaddress
        ip = ipaddress.ip_address(host)
    except ValueError:
        return True
    return not (
        ip.is_private or ip.is_loopback or ip.is_link_local
        or ip.is_reserved or ip.is_multicast
    )


def normalize_url(url: str) -> str:
    parsed = urlparse(url)
    query = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if not key.lower().startswith("utm_") and key.lower() not in {"ref", "source"}
    ]
    path = parsed.path.rstrip("/") or "/"
    return urlunparse((
        parsed.scheme.lower(),
        (parsed.netloc or "").lower(),
        path,
        "",
        urlencode(query),
        "",
    ))


def domain_of(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def rank_hit(hit: dict, topic: str, topic_terms: set[str], index: int) -> float:
    title = (hit.get("title") or "").lower()
    snippet = (hit.get("snippet") or "").lower()
    url = (hit.get("url") or "").lower()
    score = 0.0
    score += sum(2.1 for term in topic_terms if term in title)
    score += sum(0.7 for term in topic_terms if term in snippet)
    score += sum(1.4 for term in topic_terms if term in url)
    if topic and topic.lower() in title:
        score += 2.4
    if _officialish(url):
        score += 2.2
    if any(bad in url for bad in DOWNRANK_HOSTS):
        score -= 4.0
    score -= index * 0.18
    return score


def diversify(ranked: list[dict], limit: int = 6) -> list[dict]:
    picked: list[dict] = []
    seen: dict[str, int] = defaultdict(int)
    deferred = []
    for hit in ranked:
        if seen[hit["domain"]] >= 1:
            deferred.append(hit)
            continue
        seen[hit["domain"]] += 1
        picked.append(hit)
        if len(picked) >= limit:
            return picked
    for hit in deferred:
        picked.append(hit)
        if len(picked) >= limit:
            break
    return picked


def is_official_url(url: str) -> bool:
    """True for docs, developer, government, and university pages."""
    return _officialish(url)


def _officialish(url: str) -> bool:
    lowered = url.lower()
    return any(token in lowered for token in ("docs.", "/docs", "/documentation", "developer.", ".gov", ".edu"))


def _skip_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    if path.endswith(SKIP_EXTENSIONS):
        return True
    host = domain_of(url)
    if host.endswith("duckduckgo.com") or host.endswith("google.com"):
        return True
    return False


def _one_line(text: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(text or "")).strip()


def _clip(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    stop = cut.rfind(". ")
    if stop > limit * 0.6:
        return cut[: stop + 1]
    return cut
