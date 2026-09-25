"""Folio's desk. One page, one research route."""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from folio import __version__
from folio.agent import mermaid, stream_research
from folio.calling import DECLARATIONS
from folio.loop import stream_tool_loop
from folio.tools import fetch_page, search_web, summarize

ROOT = Path(__file__).resolve().parent
app = FastAPI(title="Folio", version=__version__)
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/favicon.ico")
def favicon():
    return FileResponse(ROOT / "static" / "favicon.svg", media_type="image/svg+xml")


@app.get("/api/health")
def health():
    return {"ok": True, "name": "folio", "version": __version__, "framework": "langgraph"}


@app.get("/api/graph")
def graph_view():
    return {
        "framework": "langgraph",
        "version": __version__,
        "mermaid": mermaid(),
        "tools": [
            {"name": fn.__name__, "doc": " ".join((fn.__doc__ or "").split())}
            for fn in (search_web, fetch_page, summarize)
        ],
        "function_declarations": DECLARATIONS,
        "calling": (
            "The model returns a functionCall. The app runs the tool and sends back a functionResponse. "
            "Folio's graph makes those calls itself, in that same shape."
        ),
    }


@app.get("/api/briefs")
def list_briefs():
    folder = ROOT / "briefs"
    items = []
    if folder.exists():
        for path in sorted(folder.glob("*.json")):
            try:
                data = json.loads(path.read_text())
            except Exception:
                continue
            items.append({
                "id": path.stem,
                "topic": data.get("topic") or path.stem,
                "query": data.get("query") or "",
            })
    return {"briefs": items}


@app.get("/api/briefs/{brief_id}")
def get_brief(brief_id: str):
    folder = (ROOT / "briefs").resolve()
    path = (folder / f"{brief_id}.json").resolve()
    if path.parent != folder or not path.is_file():
        return JSONResponse({"error": "No brief by that name."}, status_code=404)
    return json.loads(path.read_text())


@app.post("/api/research")
async def research(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Send a JSON body with a query."}, status_code=400)
    query = " ".join(str(body.get("query", "")).split())
    if not query:
        return JSONResponse({"error": "Ask a question first."}, status_code=400)
    if len(query) > 500:
        return JSONResponse({"error": "Keep the question under 500 characters."}, status_code=400)
    mode = body.get("mode") if body.get("mode") in {"graph", "loop"} else "graph"
    run = stream_tool_loop if mode == "loop" else stream_research

    async def events():
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def worker():
            try:
                for event in run(query):
                    loop.call_soon_threadsafe(queue.put_nowait, event)
            except Exception as exc:
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    {"type": "error", "message": f"{type(exc).__name__}: {exc}"},
                )
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        threading.Thread(target=worker, daemon=True).start()
        while True:
            event = await queue.get()
            if event is None:
                break
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
