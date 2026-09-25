const form = document.getElementById("ask");
const queryEl = document.getElementById("query");
const goBtn = document.getElementById("go");
const desk = document.getElementById("desk");
const callsEl = document.getElementById("calls");
const historyEl = document.getElementById("history");
const nodes = ["agent", "search", "read", "summarize", "answer"];

const session = [];
let running = false;
let timer = null;

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const query = queryEl.value.trim();
  if (!query || running) return;
  run(query);
});

queryEl.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
    event.preventDefault();
    form.requestSubmit();
  }
});

document.querySelectorAll(".chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    queryEl.value = chip.dataset.q;
    form.requestSubmit();
  });
});

desk.addEventListener("click", (event) => {
  const link = event.target.closest(".cites a");
  if (!link) return;
  const target = document.querySelector(link.getAttribute("href"));
  if (!target) return;
  target.classList.remove("flash");
  void target.offsetWidth;
  target.classList.add("flash");
});

async function run(query) {
  running = true;
  setBusy(true);
  resetGraph();
  callsEl.innerHTML = "";
  showWorking(query);
  activate("agent");
  const started = performance.now();
  timer = setInterval(() => {
    const seconds = Math.max(1, Math.round((performance.now() - started) / 1000));
    goBtn.textContent = `Researching · ${seconds}s`;
  }, 400);

  let gotAnswer = false;
  const controller = new AbortController();
  const abortTimer = setTimeout(() => controller.abort(), 90000);

  try {
    const response = await fetch("/api/research", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "text/event-stream",
      },
      body: JSON.stringify({
      query,
      mode: document.querySelector("input[name=mode]:checked")?.value || "graph",
    }),
      signal: controller.signal,
    });
    if (!response.ok || !response.body) {
      const text = await response.text();
      throw new Error(text || "The clerk could not start.");
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split("\n\n");
      buffer = parts.pop() || "";
      for (const part of parts) {
        const data = part
          .split("\n")
          .filter((line) => line.startsWith("data:"))
          .map((line) => line.slice(5).trim())
          .join("");
        if (!data) continue;
        const event = JSON.parse(data);
        if (event.type === "step") {
          onStep(event);
        } else if (event.type === "answer") {
          gotAnswer = true;
          onStep(event);
          renderBrief(event.answer, query);
          remember(query, event.answer);
        } else if (event.type === "error") {
          throw new Error(event.message || "The run failed.");
        }
      }
    }
    if (!gotAnswer) throw new Error("The run ended before a brief was written.");
  } catch (error) {
    const message = error.name === "AbortError"
      ? "That took too long. Try again."
      : (error.message || "The connection dropped.");
    fail(message);
  } finally {
    clearTimeout(abortTimer);
    clearInterval(timer);
    running = false;
    setBusy(false);
    goBtn.textContent = "Research";
  }
}

function onStep(event) {
  const trace = event.trace || {};
  const node = event.node || trace.node;
  if (!node) return;
  markDone(node, trace);
  const index = nodes.indexOf(node);
  if (index >= 0 && index < nodes.length - 1 && event.type !== "answer") {
    activate(nodes[index + 1]);
  }
  const status = document.getElementById("work-status");
  if (status && trace.message) status.textContent = trace.message;
  if (node === "agent") {
    const topic = document.getElementById("work-topic");
    const why = document.getElementById("work-why");
    if (topic) topic.textContent = trace.topic || topic.textContent;
    if (why) why.textContent = trace.message || "";
  }
  (trace.calls || []).forEach(renderCall);
}

function renderCall(call) {
  logCall(call);
  const slip = document.createElement("article");
  slip.className = "slip";
  const ms = typeof call.ms === "number" ? `${(call.ms / 1000).toFixed(1)}s` : "";
  let body = "";
  if (call.tool === "search_web") {
    const titles = (call.results || []).slice(0, 3).map((hit) => esc(hit.title)).join(" · ");
    body = `<p>query · ${esc(call.input?.query || "")}</p>`;
    if (call.error) body += `<p class="err">${esc(call.error)}</p>`;
    else body += `<p>${call.count || 0} results${titles ? " · " + titles : ""}</p>`;
  } else if (call.tool === "fetch_page") {
    body = `<p>${esc(call.title || call.input?.url || "")}</p>`;
    body += call.ok
      ? `<p>${Number(call.words || 0).toLocaleString()} words</p>`
      : `<p class="err">${esc(call.error || "Could not read")}</p>`;
  } else if (call.tool === "summarize") {
    body = `<p>${call.count || 0} claims from ${call.input?.documents || 0} documents</p>`;
  }
  slip.innerHTML = `<header><b>${esc(call.tool || "tool")}()</b><span>${ms}</span></header>${body}`;
  callsEl.appendChild(slip);
}

function logCall(call) {
  const box = document.getElementById("protocol");
  const log = document.getElementById("protocol-log");
  if (!box || !log || !call.tool) return;
  box.hidden = false;
  box.open = true;
  const id = `call-${log.childElementCount + 1}`;
  const args = call.input || {};
  const response = {};
  if (call.tool === "search_web") response.count = call.count || 0;
  if (call.tool === "fetch_page") {
    response.ok = Boolean(call.ok);
    response.words = call.words || 0;
  }
  if (call.tool === "summarize") response.claims = call.count || 0;
  if (call.error) response.error = call.error;
  const block = document.createElement("div");
  block.textContent = `functionCall ${id}\n${call.tool} ${JSON.stringify(args)}\nfunctionResponse ${id}\n${JSON.stringify(response)}`;
  log.appendChild(block);
}

function showWorking(query) {
  desk.innerHTML = `
    <div class="working">
      <p class="kicker">Researching</p>
      <h2 id="work-topic">${esc(query)}</h2>
      <p class="why" id="work-why">Planning the searches.</p>
      <p class="status" id="work-status" aria-live="polite">Agent is reading the question.</p>
      <div class="skeleton" aria-hidden="true">
        <span></span><span></span><span></span><span class="short"></span>
      </div>
    </div>`;
}

function renderBrief(answer, query) {
  if (!answer) return;
  if (answer.failed) {
    desk.innerHTML = `
      <article class="brief">
        <p class="kicker">${esc(answer.kicker || "Nothing to cite")}</p>
        <h2 class="topic">${esc(answer.topic || "No results")}</h2>
        <p class="lead">${esc(answer.lead?.text || "No pages came back.")}</p>
        <p class="note">${esc(answer.note || "")}</p>
      </article>`;
    return;
  }
  const topicClass = (answer.topic || "").length > 42 ? "topic long" : "topic";
  const points = (answer.points || []).map((point) =>
    `<li>${esc(point.text)}${cites(point.sources)}</li>`
  ).join("");
  const quoteSource = answer.quote
    ? (answer.sources || []).find((source) => source.id === answer.quote.sources?.[0])
    : null;
  const quote = answer.quote
    ? `<blockquote class="pull">
         <p>${esc(answer.quote.text)}${cites(answer.quote.sources)}</p>
         <footer>${esc(quoteSource?.domain || "From the pages")}</footer>
       </blockquote>`
    : "";
  const sources = (answer.sources || []).map((source) => {
    const href = safeUrl(source.url);
    const words = source.ok
      ? `${Number(source.words || 0).toLocaleString()} words`
      : "snippet only";
    return `
      <article class="source${source.cited ? "" : " uncited"}" id="src-${Number(source.id)}">
        <div class="src-id">${Number(source.id)}</div>
        <div>
          <h4>${esc(source.title)}</h4>
          <p class="meta">${esc(source.domain)} · ${words}${source.cited ? "" : " · opened, not cited"}</p>
          <p class="snippet">${esc(source.snippet || "")}</p>
          ${href ? `<a class="open" href="${href}" target="_blank" rel="noopener noreferrer">Open page</a>` : ""}
        </div>
      </article>`;
  }).join("");
  const stats = answer.stats || {};
  desk.innerHTML = `
    <article class="brief">
      <header class="brief-head">
        <div class="file">Folio · ${esc(answer.date || "")} · ${Number(stats.read || 0)} pages read</div>
        <span class="brief-actions"><button type="button" id="deeper">Go deeper</button><button type="button" id="copy">Copy brief</button></span>
      </header>
      <p class="kicker">${esc(answer.kicker || "Research brief")}</p>
      <h2 class="${topicClass}">${esc(answer.topic || query)}</h2>
      <p class="lead">${esc(answer.lead?.text || "")}${cites(answer.lead?.sources)}</p>
      ${points ? `<h3 class="points-label">${esc(answer.points_label || "What it does")}</h3><ul class="points">${points}</ul>` : ""}
      ${quote}
      <section class="sources">
        <h3>Sources</h3>
        ${sources}
      </section>
      <p class="note">${esc(answer.note || "")}</p>
    </article>`;
  const copyBtn = document.getElementById("copy");
  copyBtn.addEventListener("click", () => copyBrief(answer, copyBtn));
  const deeperBtn = document.getElementById("deeper");
  if (deeperBtn) {
    deeperBtn.addEventListener("click", () => {
      const topic = answer.topic || query;
      queryEl.value = `What is ${topic}?`;
      form.requestSubmit();
    });
  }
}

function remember(query, answer) {
  session.unshift({ query, answer });
  if (session.length > 4) session.pop();
  if (session.length < 2) {
    historyEl.hidden = true;
    return;
  }
  historyEl.hidden = false;
  historyEl.innerHTML = `<p>Earlier</p>` + session.slice(1).map((item, index) =>
    `<button type="button" data-i="${index + 1}">${esc(item.query)}</button>`
  ).join("");
  historyEl.querySelectorAll("button").forEach((button) => {
    button.addEventListener("click", () => {
      const item = session[Number(button.dataset.i)];
      if (!item || running) return;
      queryEl.value = item.query;
      callsEl.innerHTML = "";
      resetGraph();
      nodes.forEach((node) => {
        const row = document.querySelector(`[data-node="${node}"]`);
        if (row) row.classList.add("done");
      });
      renderBrief(item.answer, item.query);
    });
  });
}

async function copyBrief(answer, button) {
  const text = answer.plain || "";
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    const area = document.createElement("textarea");
    area.value = text;
    document.body.appendChild(area);
    area.select();
    document.execCommand("copy");
    area.remove();
  }
  button.textContent = "Copied";
  setTimeout(() => { button.textContent = "Copy brief"; }, 1400);
}

function resetGraph() {
  const log = document.getElementById("protocol-log");
  if (log) log.replaceChildren();
  document.querySelectorAll(".graph li").forEach((item) => {
    item.classList.remove("done", "active", "failed");
    const note = item.querySelector(".note");
    if (note) note.textContent = "";
  });
}

function activate(node) {
  document.querySelectorAll(".graph li.active").forEach((item) => item.classList.remove("active"));
  const item = document.querySelector(`[data-node="${node}"]`);
  if (item) item.classList.add("active");
}

function markDone(node, trace) {
  const item = document.querySelector(`[data-node="${node}"]`);
  if (!item) return;
  item.classList.remove("active");
  item.classList.add("done");
  const note = item.querySelector(".note");
  if (!note) return;
  if (trace.queries) {
    note.textContent = trace.queries.map((query) => `“${query}”`).join("   ");
  } else if (trace.message && trace.message !== "Restored") {
    note.textContent = trace.message;
  }
}

function fail(message) {
  document.querySelectorAll(".graph li.active").forEach((item) => {
    item.classList.remove("active");
    item.classList.add("failed");
  });
  desk.innerHTML = `
    <article class="brief">
      <p class="kicker">Stopped</p>
      <h2 class="topic">The run didn't finish</h2>
      <p class="lead">${esc(message)}</p>
    </article>`;
}

function setBusy(busy) {
  goBtn.disabled = busy;
  document.querySelectorAll(".chip").forEach((chip) => { chip.disabled = busy; });
  desk.setAttribute("aria-busy", busy ? "true" : "false");
}

function cites(ids) {
  if (!ids || !ids.length) return "";
  const links = ids
    .map((id) => `<a href="#src-${Number(id)}">${Number(id)}</a>`)
    .join("");
  return `<sup class="cites">${links}</sup>`;
}

function safeUrl(value) {
  try {
    const url = new URL(value);
    if (url.protocol === "http:" || url.protocol === "https:") return esc(url.href);
  } catch { /* ignore */ }
  return "";
}

function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  }[ch]));
}

loadShelf();

async function loadShelf() {
  const shelf = document.getElementById("shelf");
  if (!shelf) return;
  try {
    const response = await fetch("/api/briefs");
    const data = await response.json();
    const briefs = data.briefs || [];
    if (!briefs.length) {
      shelf.hidden = true;
      return;
    }
    shelf.hidden = false;
    shelf.innerHTML = `<p>On the desk</p>` + briefs.map((item) =>
      `<button type="button" data-brief="${esc(item.id)}">${esc(item.topic)}</button>`
    ).join("");
    shelf.querySelectorAll("button").forEach((button) => {
      button.addEventListener("click", () => openBrief(button.dataset.brief));
    });
  } catch {
    shelf.hidden = true;
  }
}

async function openBrief(id) {
  if (running || !id) return;
  const response = await fetch(`/api/briefs/${encodeURIComponent(id)}`);
  if (!response.ok) return;
  const answer = await response.json();
  if (answer.query) queryEl.value = answer.query;
  callsEl.innerHTML = "";
  resetGraph();
  nodes.forEach((node) => {
    const row = document.querySelector(`[data-node="${node}"]`);
    if (row) row.classList.add("done");
  });
  renderBrief(answer, answer.query || answer.topic || "");
}

const starter = queryEl.value.trim();
if (starter) run(starter);

const build = document.getElementById("build");
if (build) {
  build.addEventListener("toggle", async () => {
    if (!build.open || build.dataset.loaded) return;
    build.dataset.loaded = "1";
    const slot = document.getElementById("graph-src");
    try {
      const response = await fetch("/api/graph");
      const data = await response.json();
      const tools = (data.tools || []).map((tool) =>
        `<div class="tool-doc"><code>${esc(tool.name)}()</code> — ${esc(tool.doc)}</div>`
      ).join("");
      slot.innerHTML = `${tools}<pre>${esc(data.mermaid || "")}</pre>`;
    } catch {
      slot.textContent = "The graph description didn't load.";
    }
  });
}
