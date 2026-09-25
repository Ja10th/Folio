# Folio

A personal research agent with three tools and one path.

```
User
 ↓
Agent            plans two searches
 ↓
Search           search_web()
 ↓
Read pages       fetch_page()
 ↓
Summarize        summarize()
 ↓
Answer           with sources
```

The path is fixed on purpose. This is not a five-agent company.

## Run

```bash
pip install -r requirements.txt
python server.py
```

Open the page and leave the example in the box, or ask from the terminal:

```bash
python cli.py "Research LangGraph and tell me what it does."
```

## Where the code is

- `folio/agent.py` — the LangGraph. Start here.
- `folio/tools.py` — `search_web`, `fetch_page`, `summarize`.
- `server.py` — the desk UI.
- `static/` — the page.

No model API key. `summarize()` keeps sentences from the pages it read and numbers the sources. It does not add claims from memory.

If a later version should choose its own tools, bind these three to a model and replace the fixed edges with a tool-calling loop. Don't start there.

## Host it

Use Vercel. Netlify cannot run this app.

Netlify Functions are TypeScript, JavaScript, and Go only. Python is available while Netlify builds a site, not as a request handler. Folio is a FastAPI server, so there is nothing for Netlify to execute. A redirect from a Netlify site to a Vercel URL is a proxy, not hosting.

Vercel runs the FastAPI `app` in `server.py` as one Python function. A research call usually finishes in under 15 seconds. The first call after idle is slower, because the function has to start and import LangGraph. `vercel.json` allows 120 seconds, inside the current [function duration limit](https://vercel.com/docs/functions/limitations).

The folder you deploy must be the one that contains `server.py`, `requirements.txt`, and `pyproject.toml`. If this directory is not the Git root, set that folder as the project root in Vercel.

```bash
cd folio
npx vercel
```

Or, without the CLI: push the folder to GitHub, import the repo at [vercel.com/new](https://vercel.com/new), and deploy. Vercel reads `[tool.vercel] entrypoint = "server:app"` and installs `requirements.txt`. No build command.

Local use is unchanged: `python server.py`.
# Folio
