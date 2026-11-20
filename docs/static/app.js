"use strict";
const $ = (s, r = document) => r.querySelector(s), $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const fmtN = n => n == null ? "—" : Number(n).toLocaleString();
const fmtT = s => { s = Math.max(0, Math.floor(s)); const h = Math.floor(s / 3600), m = Math.floor(s % 3600 / 60), x = s % 60; return (h ? h + ":" : "") + String(m).padStart(2, "0") + ":" + String(x).padStart(2, "0") };
function toast(msg, err) { const t = $("#toast"); t.textContent = msg; t.className = "toast show" + (err ? " err" : ""); clearTimeout(t._h); t._h = setTimeout(() => t.className = "toast", 4200) }

const STATIC_MSG = "This is a read-only showcase. Running the pipeline needs the local app.";
function staticApiPath(u) {
  const raw = u.startsWith("http") ? new URL(u).pathname + (new URL(u).search) : u;
  const [path, qs] = raw.split("?");
  const q = new URLSearchParams(qs || "");
  const dir = (q.get("dir") || "outputs").replace(/\\/g, "/");
  const key = dir.replace(/\//g, "-");
  if (path === "/api/status") return "api/status.json";
  if (path === "/api/pipeline/facts") return "api/pipeline/facts.json";
  if (path === "/api/pipeline/flags") return "api/pipeline/flags.json";
  if (path === "/api/stats/mf") return "api/stats/mf.json";
  if (path === "/api/stats/sam") return "api/stats/sam.json";
  if (path === "/api/layers") return "api/layers-" + key + ".json";
  if (path === "/api/outputs/dirs") return "api/dirs.json";
  if (path === "/api/boxes") return "api/boxes.json";
  if (path === "/api/library") return "api/library.json";
  if (path === "/api/l2b/summary") return "api/l2b/summary.json";
  if (path === "/api/l2b/ids.bin") return "api/l2b/ids.bin";
  if (path === "/api/runs") return "api/runs.json";
  if (path === "/api/jobs/current") return "api/jobs-current.json";
  const log = path.match(/^\/api\/runs\/([^/]+)\/log$/);
  if (log) return "api/runs/" + log[1] + ".log";
  if (path === "/api/pixel") return "api/pixel-grid.json";
  if (path === "/api/spectrum") return null;
  return null;
}
async function liveFetch(u, o) {
  const r = await fetch(u, o);
  if (!r.ok) { let m = r.statusText; try { m = (await r.json()).detail || m } catch (e) { } throw new Error(m) }
  const ct = r.headers.get("content-type") || "";
  return ct.includes("json") ? r.json() : r;
}
const api = async (u, o) => {
  if (window.STATIC) {
    if (o && o.method && o.method.toUpperCase() !== "GET") throw new Error(STATIC_MSG);
    const p = staticApiPath(u);
    if (!p) throw new Error(u.startsWith("/api/spectrum") ? "Pixel spectra are available in the local app." : STATIC_MSG);
    if (p === "api/pixel-grid.json") {
      const q = new URLSearchParams((u.split("?")[1] || ""));
      return pixelFromGrid(+q.get("row"), +q.get("col"), (q.get("dir") || "outputs").replace(/\\/g, "/"));
    }
    const r = await fetch(p);
    if (!r.ok) throw new Error("Not available in the static showcase");
    const ct = r.headers.get("content-type") || "";
    if (p.endsWith(".json") || ct.includes("json")) return r.json();
    return r;
  }
  return liveFetch(u, o);
};
const fileUrl = (p, dl) => {
  const rel = String(p || "").replace(/\\/g, "/");
  if (window.STATIC) return "files/" + rel.replace(/^\//, "");
  return `/api/file?path=${encodeURIComponent(p)}${dl ? "&download=1" : ""}&t=${Date.now() % 1e7}`;
};
window._renderIndex = null;
async function renderIndex() {
  if (window._renderIndex) return window._renderIndex;
  try { window._renderIndex = await (await fetch("api/renders.json")).json(); } catch (e) { window._renderIndex = {}; }
  return window._renderIndex;
}
function nearestStretch(lo) {
  const loN = Number(lo); const opts = [0, 2, 5, 10];
  return opts.reduce((a, b) => Math.abs(b - loN) < Math.abs(a - loN) ? b : a, 2);
}
function renderUrl(params) {
  const q = params instanceof URLSearchParams ? params : new URLSearchParams(params);
  if (!window.STATIC) return "/api/render?" + q;
  const path = (q.get("path") || "").replace(/\\/g, "/");
  const lo = nearestStretch(q.get("lo") || 2);
  const hi = 100 - lo;
  const cmap = q.get("cmap") || "viridis";
  const comp = q.get("component");
  const r = q.get("r") || "0", g = q.get("g") || "1", b = q.get("b") || "2";
  const stem = path.replace(/\//g, "__");
  let name;
  if (comp != null && comp !== "") name = `${stem}__c${comp}__${cmap}__${lo}-${hi}.png`;
  else name = `${stem}__rgb${r}${g}${b}__${cmap}__${lo}-${hi}.png`;
  return { url: "renders/" + name, path, lo, hi, cmap, comp, fallback: figureFallback(path) };
}
function figureFallback(path) {
  const rel = (path || "").replace(/\\/g, "/");
  const stem = rel.split("/").pop().replace(/\.npy$/, "");
  if (!stem) return null;
  if (stem === "pca_scores") return "files/outputs/pca_rgb.png";
  if (stem === "mnf_scores") return "files/outputs/mnf_rgb.png";
  return "files/" + rel.replace(/\/maps\//, "/").replace(/\.npy$/, ".png");
}
let _pixelGrid = null;
async function pixelFromGrid(row, col, dir) {
  if (!_pixelGrid) {
    const r = await fetch("api/pixel-grid.json");
    if (!r.ok) throw new Error("no pixel grid");
    _pixelGrid = await r.json();
  }
  const G = _pixelGrid, step = G.step || 80;
  const ri = Math.max(0, Math.min(G.rows.length - 1, Math.round(row / step)));
  const ci = Math.max(0, Math.min(G.cols.length - 1, Math.round(col / step)));
  const lab = G.labels?.[ri]?.[ci];
  const ang = G.angles?.[ri]?.[ci];
  const name = (lab >= 0 && G.names && G.names[lab]) ? G.names[lab] : (lab === -1 ? "not mapped / > 0.12 rad" : lab);
  const values = [];
  if (lab != null) values.push({ layer: "outputs/maps/sam_labels.npy", label: "SAM · Labels", value: lab, name });
  if (ang != null) values.push({ layer: "outputs/maps/sam_angles.npy", label: "SAM · Angles (rad)", value: ang });
  const in_boxes = [];
  if (window._boxes) window._boxes.forEach((b, i) => { if (b[0] <= row && row < b[1] && b[2] <= col && col < b[3]) in_boxes.push(i); });
  return { row, col, values, in_boxes, snapped: [G.rows[ri], G.cols[ci]] };
}

function applyStaticUI() {
  document.body.classList.add("static");
  const note = document.getElementById("staticNote");
  if (note) note.hidden = false;
  const foot = document.getElementById("siteFoot");
  if (foot) foot.textContent = "Read-only showcase · github.com/kevinhaoaus/EMITOphiolite · run the pipeline locally to generate new maps";
  const pill = document.getElementById("jobPillText");
  if (pill) pill.textContent = "showcase";
  document.getElementById("jobPill")?.classList.add("success");
  $$(".tile").forEach(t => { t.classList.add("disabled"); t.onclick = ev => { ev.preventDefault(); toast(STATIC_MSG, true) }; t.onmouseenter = null; });
  const stop = document.getElementById("stopBtn");
  if (stop) stop.disabled = true;
  const phase = document.getElementById("phase");
  if (phase) phase.textContent = "Read-only showcase of the completed 013 scene run. Use the local app to launch a new job.";
}

async function detectStatic() {
  if (window.STATIC) return true;
  try {
    const r = await fetch("/api/status", { cache: "no-store" });
    if (r.ok) return false;
  } catch (e) { }
  try {
    const r = await fetch("api/status.json", { cache: "no-store" });
    if (r.ok) { window.STATIC = true; return true; }
  } catch (e) { }
  return false;
}

/* ---------------- navigation ---------------- */
const pages = { run: null, pipeline: () => Pipeline.init(), scores: () => Scores.init(), explorer: () => Explorer.init(), l2b: () => L2B.init(), runs: () => Runs.load() };
function go(p) { document.body.dataset.page = p; $$("nav a").forEach(a => a.classList.toggle("active", a.dataset.page === p)); $$(".page").forEach(s => s.classList.toggle("active", s.id === "page-" + p)); location.hash = p; pages[p] && pages[p](); setTimeout(() => window.dispatchEvent(new Event("resize")), 30) }
$$("nav a").forEach(a => a.onclick = () => go(a.dataset.page));
document.addEventListener("click", e => { const g = e.target.closest("[data-goto]"); if (g) go(g.dataset.goto) });

/* ---------------- status ---------------- */
async function loadStatus() {
  try {
    const s = await api("/api/status"); const sc = s.scene || {};
    $("#stats").innerHTML = [
      [sc.shape ? sc.shape.slice(0, 2).join(" × ") : "—", "scene px"], [sc.shape ? sc.shape[2] : "—", "bands"],
      [sc.good_bands ?? "—", "good bands"], [s.boxes, "exclude boxes"],
      [s.outputs.reduce((a, d) => a + d.png, 0), "PNG outputs"]].map(([b, l]) => `<div class="stat"><b>${b}</b><span>${l}</span></div>`).join("");
    $("#pyver").textContent = (window.STATIC ? "GitHub Pages · read-only" : "server · python " + s.python);
    if (s.job && s.job.state === "running" && !Run.job) Run.attach(s.job);
  } catch (e) { $("#stats").innerHTML = `<div class="stat"><b>offline</b><span>${esc(e.message)}</span></div>` }
}

/* ---------------- RUN ---------------- */
const Run = {
  job: null, es: null, flags: null, timer: null,
  params() {
    const extra = {}; $$("#advFlags [data-flag]").forEach(i => { const v = i.type === "checkbox" ? i.checked : i.value.trim(); if (v) extra[i.dataset.flag] = v });
    return { components: +$("#pComponents").value || null, max_fit_pixels: +$("#pMaxFit").value || null, chunk_rows: +$("#pChunk").value || null,
      smoke_rows: +$("#pSmokeRows").value || 256, mask_boxes: $("#pBoxes").checked, out: $("#pOut").value.trim() || null, extra };
  },
  async loadFlags() {
    try {
      const f = await api("/api/pipeline/flags"); this.flags = f;
      const known = ["--reflectance", "--usgs", "--out", "--mask-boxes", "--components", "--max-fit-pixels", "--chunk-rows", f.subset_flag];
      const names = f.flags.map(x => x.flag);
      $("#flagsInfo").innerHTML = `${names.length} flags · ${["--max-fit-pixels", "--chunk-rows"].map(n => names.includes(n) ? `<span style="color:#5E7D5A">${n} ✓</span>` : `<span style="color:#B5523B">${n} pending</span>`).join(" · ")} · subset: ${f.subset_flag ? `<span style="color:#5E7D5A">${f.subset_flag}</span>` : '<span style="color:#B5523B">none yet</span>'}`;
      const extra = f.flags.filter(x => !known.includes(x.flag));
      $("#advFlags").innerHTML = extra.length ? extra.map(x => x.is_bool ? `<label class="switch" title="${esc(x.help)}"><input type="checkbox" data-flag="${x.flag}"><span class="sl"></span>${x.flag}</label>`
        : `<label title="${esc(x.help)}">${x.flag}<input type="text" data-flag="${x.flag}" placeholder="${esc(x.default || x.metavar || "")}"></label>`).join("") : '<span class="muted small">no extra flags</span>';
      $("#helpText").textContent = f.help || f.error || "";
      $$("#advFlags input").forEach(i => i.oninput = () => this.preview("full"));
    } catch (e) { $("#flagsInfo").textContent = "could not read --help: " + e.message }
    this.preview("full");
  },
  async preview(kind) {
    if (window.STATIC) { $("#cmdPreview").textContent = "Available in the local app only."; return } try { const r = await api("/api/pipeline/preview", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ kind, params: this.params() }) }); $("#cmdPreview").textContent = r.command || ("⚠ " + r.error) } catch (e) { }
  },
  async start(kind) {
    if (window.STATIC) return toast(STATIC_MSG, true);
    if (this.job && this.job.state === "running") return toast("A job is already running — stop it first.", true);
    try { const j = await api("/api/jobs", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ kind, params: this.params() }) }); $("#term").innerHTML = ""; this.attach(j); toast("Started " + kind) }
    catch (e) { toast(e.message, true) }
  },
  attach(j) {
    this.job = j; this.es && this.es.close(); $("#term").innerHTML = `<span class="cmd">$ ${esc(j.command)}</span>\n`;
    $("#termTitle").textContent = j.id; this.render();
    const es = this.es = new EventSource(`/api/jobs/${j.id}/stream?since=0`);
    es.addEventListener("lines", e => { const d = JSON.parse(e.data); const t = $("#term"); const frag = d.lines.map(l => this.line(l)).join(""); t.insertAdjacentHTML("beforeend", frag); if ($("#autoscroll").checked) t.scrollTop = t.scrollHeight });
    es.addEventListener("status", e => { this.job = JSON.parse(e.data); this.render() });
    es.addEventListener("end", () => { es.close(); this.render(); const ok = this.job.state === "success"; toast(`${this.job.kind} ${this.job.state} (exit ${this.job.exit_code}) in ${fmtT(this.job.runtime)}`, !ok); loadStatus(); Explorer.loaded = false; Runs.loaded = false; Pipeline.loaded = false; Scores.loaded = false; L2B.maybeReload(this.job) });
    clearInterval(this.timer); this.timer = setInterval(() => this.tick(), 500);
  },
  line(l) { const t = esc(l.text); const c = /error|traceback|failed|exception/i.test(l.text) ? "err" : /warn/i.test(l.text) ? "warn" : /passed|done|->|success/i.test(l.text) ? "ok" : ""; return `<span class="ts">${fmtT(l.t)} </span><span class="${c}">${t}</span>\n` },
  tick() { const j = this.job; if (!j) return; const el = j.state === "running" ? (Date.now() / 1000 - j.started) : j.runtime; $("#elapsed").textContent = fmtT(el) },
  render() {
    const j = this.job, run = j && j.state === "running";
    $("#stopBtn").disabled = !run; $$(".tile").forEach(t => t.classList.toggle("disabled", !!run));
    const pill = $("#jobPill"); pill.className = "pill " + (j ? j.state : "idle"); $("#jobPillText").textContent = j ? `${j.kind} · ${j.state}` : "idle";
    const p = $(".progress"); const pr = j && j.progress;
    p.classList.toggle("indet", !!run && pr == null); $("#progBar").style.width = pr != null ? (pr * 100).toFixed(1) + "%" : (j && !run ? (j.state === "success" ? "100%" : "0") : "0");
    $("#progPct").textContent = pr != null ? (pr * 100).toFixed(0) + "%" : "";
    $("#phase").textContent = !j ? "No job running." : run ? (j.phase || "running…") + " · " + j.n_lines + " log lines" : `${j.state} · exit ${j.exit_code} · ${fmtT(j.runtime)} · log ${j.log}`;
    this.tick();
  },
  init() {
    $$(".tile").forEach(t => { t.onclick = () => this.start(t.dataset.kind); t.onmouseenter = () => this.preview(t.dataset.kind) });
    $("#stopBtn").onclick = async () => { if (this.job) { await api(`/api/jobs/${this.job.id}/stop`, { method: "POST" }); toast("Stopping (killing process tree)…") } };
    $$("#page-run .form input").forEach(i => i.oninput = () => { $("#smokeRowsLbl").textContent = $("#pSmokeRows").value; this.preview("full") });
    this.loadFlags();
  }
};

/* ---------------- MapView (zoom / pan canvas) ---------------- */
class MapView {
  constructor(wrap, mini, scaleEl, scaleLbl) {
    this.wrap = wrap; this.cv = $("canvas.mapcanvas", wrap); this.ctx = this.cv.getContext("2d"); this.mini = mini; this.sEl = scaleEl; this.sLbl = scaleLbl;
    this.img = null; this.s = 1; this.tx = 0; this.ty = 0; this.overlay = null; this.marker = null; this.onHover = null; this.onClick = null; this.coords = true;
    const cv = this.cv; let drag = null, moved = false;
    cv.addEventListener("wheel", e => { e.preventDefault(); const r = cv.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top, f = Math.exp(-e.deltaY * 0.0015); this.zoomAt(x, y, f) }, { passive: false });
    cv.addEventListener("mousedown", e => { drag = { x: e.clientX, y: e.clientY, tx: this.tx, ty: this.ty }; moved = false; cv.classList.add("drag") });
    window.addEventListener("mouseup", e => { if (drag && !moved && e.target === cv && this.onClick) { const p = this.toImg(e); p && this.onClick(p.r, p.c) } drag = null; cv.classList.remove("drag") });
    window.addEventListener("mousemove", e => { if (drag) { const dx = e.clientX - drag.x, dy = e.clientY - drag.y; if (Math.abs(dx) + Math.abs(dy) > 3) moved = true; this.tx = drag.tx + dx; this.ty = drag.ty + dy; this.draw() } });
    cv.addEventListener("mousemove", e => { if (drag) return; const p = this.toImg(e); this.onHover && this.onHover(p) });
    cv.addEventListener("dblclick", () => this.fit());
    mini && mini.addEventListener("click", e => { if (!this.img) return; const r = mini.getBoundingClientRect(), k = this.mk; const ix = (e.clientX - r.left - this.mo[0]) / k, iy = (e.clientY - r.top - this.mo[1]) / k; this.tx = this.cv.clientWidth / 2 - ix * this.s; this.ty = this.cv.clientHeight / 2 - iy * this.s; this.draw() });
    new ResizeObserver(() => { this.resize() }).observe(wrap);
  }
  resize() { const d = devicePixelRatio || 1, w = this.cv.clientWidth, h = this.cv.clientHeight; if (!w) return; this.cv.width = w * d; this.cv.height = h * d; if (this.img && !this._fitted) this.fit(); else this.draw() }
  set(img, keepView) { this.img = img; if (!keepView || !this._fitted) this.fit(); else this.draw() }
  fit() { if (!this.img) return this.draw(); const w = this.cv.clientWidth, h = this.cv.clientHeight; if (!w) return; this.s = Math.min(w / this.img.width, h / this.img.height) * 0.94; this.tx = (w - this.img.width * this.s) / 2; this.ty = (h - this.img.height * this.s) / 2; this._fitted = true; this.draw() }
  zoomAt(x, y, f) { const ns = Math.min(64, Math.max(0.05, this.s * f)); f = ns / this.s; this.tx = x - (x - this.tx) * f; this.ty = y - (y - this.ty) * f; this.s = ns; this.draw() }
  toImg(e) { if (!this.img) return null; const r = this.cv.getBoundingClientRect(); const c = Math.floor((e.clientX - r.left - this.tx) / this.s), rr = Math.floor((e.clientY - r.top - this.ty) / this.s); if (c < 0 || rr < 0 || c >= this.img.width || rr >= this.img.height) return null; return { r: rr, c } }
  draw() {
    const d = devicePixelRatio || 1, ctx = this.ctx; ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.clearRect(0, 0, this.cv.width, this.cv.height);
    if (!this.img) return this.drawMini();
    ctx.setTransform(d * this.s, 0, 0, d * this.s, d * this.tx, d * this.ty); ctx.imageSmoothingEnabled = this.s < 1.5;
    ctx.drawImage(this.img, 0, 0); ctx.lineWidth = 1 / this.s; ctx.strokeStyle = "#D9D5CB"; ctx.strokeRect(0, 0, this.img.width, this.img.height);
    this.overlay && this.overlay(ctx, this.s);
    if (this.marker) { const [r, c] = this.marker; ctx.lineWidth = 1.5 / this.s; ctx.strokeStyle = "#191919"; ctx.strokeRect(c - 0.5, r - 0.5, 2, 2); ctx.beginPath(); const L = 12 / this.s; ctx.moveTo(c + .5 - L, r + .5); ctx.lineTo(c + .5 - 3 / this.s, r + .5); ctx.moveTo(c + .5 + 3 / this.s, r + .5); ctx.lineTo(c + .5 + L, r + .5); ctx.moveTo(c + .5, r + .5 - L); ctx.lineTo(c + .5, r + .5 - 3 / this.s); ctx.moveTo(c + .5, r + .5 + 3 / this.s); ctx.lineTo(c + .5, r + .5 + L); ctx.stroke() }
    this.drawScale(); this.drawMini();
  }
  drawScale() {
    if (!this.sEl) return; if (!this.coords) { this.sEl.parentNode.style.display = "none"; return } this.sEl.parentNode.style.display = "";
    const mpp = 60 / this.s; let m = 110 * mpp; const p = Math.pow(10, Math.floor(Math.log10(m))); m = [1, 2, 5, 10].map(k => k * p).filter(v => v <= m).pop() || p;
    this.sEl.style.width = (m / mpp) + "px"; this.sLbl.textContent = m >= 1000 ? (m / 1000) + " km" : m + " m";
  }
  drawMini() {
    const mc = this.mini; if (!mc) return; const ctx = mc.getContext("2d"); ctx.clearRect(0, 0, mc.width, mc.height); if (!this.img) return;
    const k = this.mk = Math.min((mc.width - 12) / this.img.width, (mc.height - 12) / this.img.height); const ox = (mc.width - this.img.width * k) / 2, oy = (mc.height - this.img.height * k) / 2; this.mo = [ox, oy];
    ctx.drawImage(this.img, ox, oy, this.img.width * k, this.img.height * k);
    const x0 = -this.tx / this.s, y0 = -this.ty / this.s, w = this.cv.clientWidth / this.s, h = this.cv.clientHeight / this.s;
    ctx.strokeStyle = "#CC785C"; ctx.lineWidth = 1.2; ctx.fillStyle = "rgba(204,120,92,.12)"; ctx.fillRect(ox + x0 * k, oy + y0 * k, w * k, h * k); ctx.strokeRect(ox + x0 * k, oy + y0 * k, w * k, h * k);
  }
}
const loadImg = src => new Promise((res, rej) => { const i = new Image(); i.onload = () => res(i); i.onerror = () => rej(new Error("image failed: " + src)); i.src = src });

/* ---------------- EXPLORER ---------------- */
const CMAPS = ["viridis", "magma", "inferno", "plasma", "cividis", "turbo", "RdBu_r", "coolwarm", "gray"];
const Explorer = {
  loaded: false, view: null, dir: "outputs", layers: [], cur: null, boxes: [], opts: { lo: 2, hi: 98, cmap: "viridis", r: 0, g: 1, b: 2, component: null }, pinned: null, spec: null, mode: "cr", shown: null,
  async init() {
    if (!this.view) {
      this.view = new MapView($("#exMap"), $("#exMini"), $("#exScale"), $("#exScaleLbl"));
      this.view.overlay = (ctx, s) => { if (!$("#boxToggle").checked || !this.view.coords) return; ctx.lineWidth = 1.5 / s; this.boxes.forEach((b, i) => { ctx.fillStyle = "rgba(25,25,25,.10)"; ctx.strokeStyle = "rgba(181,82,59,.95)"; ctx.fillRect(b[2], b[0], b[3] - b[2], b[1] - b[0]); ctx.strokeRect(b[2], b[0], b[3] - b[2], b[1] - b[0]); if (s > .35) { ctx.fillStyle = "#B5523B"; ctx.font = `${11 / s}px Segoe UI`; ctx.fillText("#" + i, b[2] + 3 / s, b[0] + 12 / s) } }) };
      let hov = null;
      this.view.onHover = p => { if (!p) { $("#exReadout").textContent = "outside map"; return } if (!this.view.coords) { $("#exReadout").textContent = "figure PNG · pixel coords n/a"; return } $("#exReadout").innerHTML = `row <b>${p.r}</b> · col <b>${p.c}</b>`; clearTimeout(hov); hov = setTimeout(() => this.hover(p), 70) };
      this.view.onClick = (r, c) => { if (this.view.coords) this.pin(r, c); else toast("Pick a .npy map layer to inspect pixels and spectra") };
      api("/api/library").then(l => { Explorer.libNames = l.names }).catch(() => { });
      $("#boxToggle").onchange = () => this.view.draw(); $("#refreshLayers").onclick = () => { this.loaded = false; this.init() };
      $$("#specMode button").forEach(b => b.onclick = () => { $$("#specMode button").forEach(x => x.classList.toggle("on", x === b)); this.mode = b.dataset.m; this.plot() });
      api("/api/boxes").then(b => { this.boxes = b.boxes; window._boxes = b.boxes; this.view.draw() });
      window.addEventListener("resize", () => this.plot());
    }
    if (this.loaded) return; this.loaded = true;
    const dirs = await api("/api/outputs/dirs").catch(() => []);
    if (!dirs.find(d => d.dir === this.dir)) this.dir = dirs[0]?.dir || "outputs";
    $("#dirSeg").innerHTML = (dirs.length ? dirs : [{ dir: "outputs", png: 0, npy: 0 }]).map(d => `<button data-d="${d.dir}" class="${d.dir === this.dir ? "on" : ""}">${d.dir}<span class="tag">${d.png} png · ${d.npy} npy</span></button>`).join("");
    $$("#dirSeg button").forEach(b => b.onclick = () => { this.dir = b.dataset.d; this.loaded = false; this.init() });
    const L = await api("/api/layers?dir=" + encodeURIComponent(this.dir)).catch(() => ({ layers: [], texts: [] }));
    this.layers = L.layers.filter(l => l.type !== "error"); this.texts = L.texts;
    $("#layerCount").textContent = `${this.layers.filter(l => l.type !== "figure").length} map layers · ${this.layers.filter(l => l.type === "figure").length} figures`;
    $("#exEmpty").classList.toggle("show", !this.layers.length);
    const groups = {}; this.layers.forEach(l => (groups[l.group] = groups[l.group] || []).push(l));
    const sortKey = l => l.type === "figure" ? 1 : 0;
    $("#layerList").innerHTML = Object.entries(groups).map(([g, ls]) => `<div class="grp">${esc(g)}</div>` + ls.sort((a, b) => sortKey(a) - sortKey(b)).map(l => `<div class="ly" data-id="${esc(l.id)}">${esc(l.label)}<span class="tag ${l.type === "figure" ? "" : "npy"}">${l.type === "figure" ? "png" : l.type === "multi" ? "k=" + l.shape[2] : l.type}</span></div>`).join("")).join("") + (this.texts.length ? `<div class="grp">Text</div>` + this.texts.map(t => `<div class="ly" data-txt="${esc(t.name)}">${esc(t.name.split("/").pop())}</div>`).join("") : "");
    $$("#layerList .ly[data-id]").forEach(e => e.onclick = () => this.select(e.dataset.id));
    $$("#layerList .ly[data-txt]").forEach(e => e.onclick = () => { const t = this.texts.find(x => x.name === e.dataset.txt); $("#pxHead").textContent = t.name; $("#pxTable").innerHTML = `<pre class="help" style="grid-column:1/-1;max-height:300px">${esc(t.content)}</pre>` });
    const first = this.layers.find(l => l.type === "multi") || this.layers.find(l => l.type !== "figure") || this.layers[0];
    if (first) this.select(first.id); else { this.view.set(null); $("#ctrlBody").innerHTML = "" }
  },
  ctrls() {
    const l = this.cur, o = this.opts; if (!l) return;
    if (l.type === "figure") { $("#ctrlBody").innerHTML = `<div class="muted small">Matplotlib figure (${l.w}×${l.h}). Rendered PNG from the pipeline — pixel coordinates aren't scene rows/cols.<br><br><a href="${fileUrl(l.file, 1)}">Download PNG</a></div>`; $("#exLegend").innerHTML = ""; return }
    let h = "";
    if (l.type === "multi") { const k = l.shape[2], opt = v => [...Array(k).keys()].map(i => `<option value="${i}" ${i === v ? "selected" : ""}>C${i + 1}</option>`).join(""); h += `<label>Mode<select id="cMode"><option value="rgb">RGB composite</option>${[...Array(k).keys()].map(i => `<option value="${i}" ${o.component === i ? "selected" : ""}>Single · C${i + 1}</option>`).join("")}</select></label>`; if (o.component == null) h += `<div class="rgbsel"><label>R<select id="cR">${opt(o.r)}</select></label><label>G<select id="cG">${opt(o.g)}</select></label><label>B<select id="cB">${opt(o.b)}</select></label></div>` }
    if (l.type !== "labels") { h += `<label>Low percentile <span class="mono">${o.lo}%</span><input type="range" id="cLo" min="0" max="49" step="0.5" value="${o.lo}"></label><label>High percentile <span class="mono">${o.hi}%</span><input type="range" id="cHi" min="51" max="100" step="0.5" value="${o.hi}"></label>`; if (l.type === "scalar" || o.component != null) h += `<label>Colormap<select id="cMap">${CMAPS.map(c => `<option ${c === o.cmap ? "selected" : ""}>${c}</option>`).join("")}</select></label>` }
    if (!window.STATIC) h += `<a class="small" href="${fileUrl(l.file, 1)}">Download .npy</a>`;
    $("#ctrlBody").innerHTML = h; $("#cMode") && ($("#cMode").value = o.component == null ? "rgb" : o.component);
    const upd = () => { if ($("#cMode")) o.component = $("#cMode").value === "rgb" ? null : +$("#cMode").value; ["r", "g", "b"].forEach(c => { const e = $("#c" + c.toUpperCase()); if (e) o[c] = +e.value }); if ($("#cLo")) { o.lo = +$("#cLo").value; o.hi = +$("#cHi").value } if ($("#cMap")) o.cmap = $("#cMap").value; this.render() };
    $$("#ctrlBody select").forEach(s => s.onchange = () => { upd(); this.ctrls() });
    $$("#ctrlBody input[type=range]").forEach(s => { s.oninput = () => s.previousElementSibling.textContent = s.value + "%"; s.onchange = upd });
  },
  select(id) {
    this.cur = this.layers.find(l => l.id === id); $$("#layerList .ly").forEach(e => e.classList.toggle("on", e.dataset.id === id));
    if (this.cur.type !== "multi") this.opts.component = null; this.ctrls(); this.render(false);
  },
  async render(keep = true) {
    const l = this.cur, o = this.opts; if (!l) return;
    try {
      if (l.type === "figure") { this.view.coords = false; this.view.set(await loadImg(fileUrl(l.file)), false); $("#exLegend").innerHTML = ""; return }
      this.view.coords = true;
      const q = new URLSearchParams({ path: l.file, r: o.r, g: o.g, b: o.b, lo: o.lo, hi: o.hi, cmap: o.cmap }); if (o.component != null) q.set("component", o.component);
      let meta = {}, bmp;
      if (window.STATIC) {
        const info = renderUrl(q);
        let src = info.url;
        try { bmp = await loadImg(src); } catch (e) { if (info.fallback) { src = info.fallback; bmp = await loadImg(src); } else throw e }
        const idx = await renderIndex();
        const ent = idx[(l.file || "").replace(/\\/g, "/")] || {};
        const k = (o.component != null ? ("c" + o.component) : ("rgb" + o.r + o.g + o.b)) + "|" + (o.cmap || "viridis") + "|" + nearestStretch(o.lo) + "-" + (100 - nearestStretch(o.lo));
        meta = (ent[k] && ent[k].meta) || {};
      } else {
        const r = await fetch("/api/render?" + q); if (!r.ok) throw new Error((await r.json()).detail);
        meta = JSON.parse(r.headers.get("X-Meta") || "{}"); bmp = await createImageBitmap(await r.blob());
      }
      this.view.set(bmp, keep && this.shown === l.shape?.slice(0, 2).join()); this.shown = l.shape?.slice(0, 2).join();
      const lg = $("#exLegend");
      if (meta.labels) { lg.innerHTML = `<div class="fh" style="margin-top:8px">Labels</div>` + meta.labels.map(v => `<div class="chip" style="padding:3px 4px"><span class="sw" style="background:${meta.palette[v % meta.palette.length]}"></span><span class="nm">${v}${Explorer.libNames && Explorer.libNames[v] ? " · " + Explorer.libNames[v] : ""}</span></div>`).join("") }
      else if (meta.ranges && meta.ranges.length === 1) { const g = { viridis: "#440154,#3b528b,#21918c,#5ec962,#fde725", magma: "#000004,#51127c,#b73779,#fc8961,#fcfdbf", inferno: "#000004,#56106e,#bb3754,#f98e09,#fcffa4", plasma: "#0d0887,#7e03a8,#cc4778,#f89540,#f0f921", cividis: "#00224e,#575d6d,#a59c74,#fee838", turbo: "#30123b,#4686fb,#1ae4b6,#a2fc3c,#fabb39,#e4460a,#7a0403", RdBu_r: "#053061,#4393c3,#f7f7f7,#d6604d,#67001f", coolwarm: "#3b4cc0,#aac7fd,#f7b89c,#b40426", gray: "#000,#fff" }[o.cmap] || "#000,#fff"; lg.innerHTML = `<div class="cbar" style="background:linear-gradient(90deg,${g})"></div><div class="cbar-l"><span>${meta.ranges[0][0].toPrecision(3)}</span><span>${meta.ranges[0][1].toPrecision(3)}</span></div>` }
      else if (meta.ranges) lg.innerHTML = `<div class="cbar-l" style="flex-direction:column;gap:2px">${meta.ranges.map((r, i) => `<span style="color:${["#B5523B", "#5E7D5A", "#4F6D8A"][i]}">${"RGB"[i]} ${r[0].toPrecision(3)} … ${r[1].toPrecision(3)}</span>`).join("")}</div>`;
    } catch (e) { toast(e.message, true) }
  },
  async hover(p) {
    try { const d = await api(`/api/pixel?dir=${encodeURIComponent(this.dir)}&row=${p.r}&col=${p.c}`); const v = d.values.find(x => x.layer === this.cur?.file); const fmt = x => x == null ? "NaN" : Array.isArray(x) ? "[" + x.slice(0, 3).map(y => y == null ? "NaN" : (+y).toPrecision(3)).join(", ") + (x.length > 3 ? ", …" : "") + "]" : typeof x === "number" ? (+x).toPrecision(4) : x; $("#exReadout").innerHTML = `row <b>${p.r}</b> · col <b>${p.c}</b>${v ? " · " + fmt(v.value) : ""}${d.in_boxes.length ? ' · <span style="color:#B5523B">excluded box #' + d.in_boxes[0] + "</span>" : ""}`; if (!this.pinned) this.table(d, fmt) } catch (e) { }
  },
  table(d, fmt) { $("#pxHead").textContent = `row ${d.row} · col ${d.col}`; $("#pxTable").innerHTML = d.values.length ? d.values.map(v => `<div class="k">${esc(v.label)}</div><div class="v">${Array.isArray(v.value) ? v.value.map(y => y == null ? "NaN" : (+y).toPrecision(4)).join("  ") : fmt(v.value)}</div>`).join("") : '<div class="muted">no .npy maps in this output dir</div>' },
  async pin(r, c) {
    this.pinned = [r, c]; this.view.marker = [r, c]; this.view.draw();
    const fmt = x => x == null ? "NaN" : typeof x === "number" ? (+x).toPrecision(4) : x;
    api(`/api/pixel?dir=${encodeURIComponent(this.dir)}&row=${r}&col=${c}`).then(d => this.table(d, fmt)).catch(() => { });
    $("#specMatches").innerHTML = '<div class="spinner"></div>';
    try { this.spec = await api(`/api/spectrum?row=${r}&col=${c}`); this.sel = new Set(this.spec.matches.slice(0, 3).map(m => m.name)); this.plot() } catch (e) { $("#specMatches").innerHTML = `<span class="muted small">${esc(e.message)}</span>`; this.spec = null; this.plot() }
  },
  plot() {
    const cv = $("#specPlot"); if (!cv || !cv.clientWidth) return; const s = this.spec, cr = this.mode === "cr";
    const cols = ["#5B6770", "#7A7D5C", "#C9A66B", "#4F6D8A", "#8C5E58", "#6E8B8E", "#A3866A", "#9A6F8F", "#6B7F5E", "#B0A06A", "#7C7468", "#5E7D5A", "#A07A60"];
    const series = [];
    if (s) { const y = cr ? s.cr : s.raw; if (y) series.push({ y, color: "#CC785C", w: 2, label: `pixel (${s.row}, ${s.col})` }); s.matches.forEach((m, i) => { if (this.sel.has(m.name)) series.push({ y: cr ? m.cr : m.raw, color: cols[i % cols.length], w: 1.2, dash: [4, 3], label: m.name }) }) }
    plotLines(cv, s ? s.wavelengths : [], series, "Wavelength (nm)", cr ? "Continuum removed" : "Reflectance", s ? null : "Click a map pixel to load its spectrum");
    if (s) { $("#specMatches").innerHTML = (s.note ? `<span class="muted small">${esc(s.note)}</span>` : "") + s.matches.map((m, i) => `<span class="mchip ${this.sel.has(m.name) ? "on" : ""}" data-n="${esc(m.name)}" style="color:${cols[i % cols.length]}"><i style="background:${cols[i % cols.length]}"></i><span style="color:var(--text)">${esc(m.name)}</span><b>${m.angle == null ? "—" : m.angle.toFixed(3) + " rad"}</b></span>`).join(""); $$("#specMatches .mchip").forEach(e => e.onclick = () => { const n = e.dataset.n; this.sel.has(n) ? this.sel.delete(n) : this.sel.add(n); this.plot() }) }
  }
};
function plotLines(cv, x, series, xl, yl, emptyMsg) {
  const d = devicePixelRatio || 1, W = cv.clientWidth, H = cv.clientHeight; cv.width = W * d; cv.height = H * d; const ctx = cv.getContext("2d"); ctx.scale(d, d); ctx.clearRect(0, 0, W, H);
  const P = { l: 58, r: 14, t: 18, b: 42 }; ctx.font = "11px Segoe UI, system-ui"; ctx.fillStyle = "#6F6B60";
  if (!series.length) { ctx.textAlign = "center"; ctx.fillText(emptyMsg || "no data", W / 2, H / 2); return }
  let ys = []; series.forEach(s => s.y.forEach(v => v != null && isFinite(v) && ys.push(v))); let y0 = Math.min(...ys), y1 = Math.max(...ys); const pad = (y1 - y0) * 0.06 || 0.05; y0 -= pad; y1 += pad;
  const x0 = Math.min(...x), x1 = Math.max(...x); const X = v => P.l + (v - x0) / (x1 - x0) * (W - P.l - P.r), Y = v => H - P.b - (v - y0) / (y1 - y0) * (H - P.t - P.b);
  ctx.strokeStyle = "#E5E2DA"; ctx.lineWidth = 1; ctx.textAlign = "right";
  for (let i = 0; i <= 5; i++) { const v = y0 + (y1 - y0) * i / 5, yy = Y(v); ctx.beginPath(); ctx.moveTo(P.l, yy); ctx.lineTo(W - P.r, yy); ctx.stroke(); ctx.fillText(v.toFixed(2), P.l - 6, yy + 4) }
  ctx.textAlign = "center"; for (let v = Math.ceil(x0 / 250) * 250; v <= x1; v += 250) { const xx = X(v); ctx.beginPath(); ctx.moveTo(xx, P.t); ctx.lineTo(xx, H - P.b); ctx.stroke(); ctx.fillText(v, xx, H - P.b + 15) }
  ctx.font = "italic 13px Georgia, 'Source Serif 4', serif"; ctx.fillStyle = "#3D3929"; ctx.fillText(xl, (W + P.l) / 2, H - 6); ctx.save(); ctx.translate(13, (H - P.b) / 2); ctx.rotate(-Math.PI / 2); ctx.fillText(yl, 0, 0); ctx.restore();
  // absorption-window shading
  [[2180, 2265, "chlorite"], [2280, 2380, "carbonate"], [550, 950, "hematite"]].forEach(([a, b, n]) => { if (b < x0 || a > x1) return; ctx.fillStyle = "rgba(204,120,92,.06)"; ctx.fillRect(X(a), P.t, X(b) - X(a), H - P.t - P.b); ctx.font = "11px Segoe UI, system-ui"; ctx.fillStyle = "#A8A396"; ctx.fillText(n, (X(a) + X(b)) / 2, P.t + 10) });
  series.slice().reverse().forEach(s => {
    ctx.strokeStyle = s.color; ctx.lineWidth = s.w; ctx.setLineDash(s.dash || []); ctx.lineJoin = "round"; ctx.beginPath(); let pen = false;
    for (let i = 0; i < x.length; i++) { const v = s.y[i]; if (v == null || !isFinite(v) || (i && x[i] - x[i - 1] > 40)) { if (v == null || !isFinite(v)) { pen = false; continue } pen = false } pen ? ctx.lineTo(X(x[i]), Y(v)) : ctx.moveTo(X(x[i]), Y(v)); pen = true }
    ctx.stroke(); ctx.setLineDash([]);
  });
}

/* ---------------- L2B ---------------- */
const L2B = {
  loaded: false, view: null, sum: null, ids: null, vis: new Set(), hl: null,
  async init() {
    if (!this.view) {
      this.view = new MapView($("#l2bMap"), $("#l2bMini"), $("#l2bScale"), $("#l2bScaleLbl"));
      this.view.onHover = p => { if (!p || !this.ids) { $("#l2bReadout").textContent = "outside map"; return } const id = this.ids[p.r * this.sum.shape[1] + p.c]; const c = this.byId[id]; $("#l2bReadout").innerHTML = `row <b>${p.r}</b> · col <b>${p.c}</b> · id <b>${id}</b> ${c ? "· " + esc(c.name) : id === this.sum.fill ? "· fill" : ""}`; if (this.hl !== id) { this.hl = id; $$("#l2bLegend .chip").forEach(e => e.classList.toggle("hl", +e.dataset.id === id)) } };
      $("#l2bAll").onclick = () => { this.sum.classes.forEach(c => c.id && this.vis.add(c.id)); this.paint() };
      $("#l2bNone").onclick = () => { this.vis.clear(); this.paint() };
      $("#l2bTop").onclick = () => { this.vis = new Set(this.sum.classes.filter(c => c.top12 && c.id).map(c => c.id)); this.paint() };
      $("#l2bShow0").onchange = () => this.paint();
    }
    if (this.loaded) return; this.loaded = true; $("#l2bEmpty").classList.add("show");
    try {
      const [s, buf] = await Promise.all([api("/api/l2b/summary"), fetch(window.STATIC ? "api/l2b/ids.bin" : "/api/l2b/ids.bin").then(r => { if (!r.ok) throw new Error("L2B ids missing"); return r.arrayBuffer(); })]);
      this.sum = s; this.recolor(s); this.ids = new Int16Array(buf); this.byId = {}; s.classes.forEach(c => this.byId[c.id] = c);
      this.vis = new Set(s.classes.filter(c => c.id).map(c => c.id));
      this.cv = document.createElement("canvas"); this.cv.width = s.shape[1]; this.cv.height = s.shape[0];
      $("#l2bInfo").innerHTML = `${esc(s.file)}<br>${s.shape.join(" × ")} · ${s.n_ids} ids present · ${fmtN(s.valid_pixels)} valid px · fill ${s.fill} × ${fmtN(s.fill_count)} · ${s.n_metadata} metadata names`;
      const tot = s.valid_pixels, mx = Math.max(...s.classes.filter(c => c.id).map(c => c.count));
      $("#l2bTable").innerHTML = `<tr><th></th><th>id</th><th>mineral (index == id)</th><th>group</th><th>pixels</th><th>%</th><th></th></tr>` + s.classes.map(c => `<tr><td><span class="sw" style="display:inline-block;width:12px;height:12px;border-radius:3px;background:${c.color}"></span></td><td class="mono">${c.id}</td><td>${esc(c.name)}${c.named ? "" : ' <span class="tag">no name</span>'}</td><td class="mono">${c.group ?? ""}</td><td class="mono">${fmtN(c.count)}</td><td class="mono">${(100 * c.count / tot).toFixed(c.id ? 3 : 1)}</td><td class="barcell">${c.id ? `<div class="b" style="width:${(100 * c.count / mx).toFixed(1)}%"></div>` : ""}</td></tr>`).join("");
      this.paint(); this.view.set(this.cv); $("#l2bEmpty").classList.remove("show");
    } catch (e) { $("#l2bEmpty").innerHTML = `<h3>Couldn't load L2B</h3><p>${esc(e.message)}</p>` }
  },
  recolor(s) {
    const base = ["#CC785C", "#5B6770", "#7A7D5C", "#C9A66B", "#8C5E58", "#6E8B8E", "#A3866A", "#4F6D8A", "#B0A06A", "#9A6F8F", "#6B7F5E", "#D4A28A"];
    const hsl = (h, s, l) => { const k = n => (n + h / 30) % 12, a = s * Math.min(l, 1 - l), f = n => l - a * Math.max(-1, Math.min(k(n) - 3, 9 - k(n), 1)); return "#" + [f(0), f(8), f(4)].map(x => Math.round(x * 255).toString(16).padStart(2, "0")).join("") };
    let i = 0; s.classes.forEach(c => { if (c.id === 0) { c.color = "#D6D2C7"; return } c.color = i < base.length ? base[i] : hsl((i * 137.5) % 360, 0.22, 0.55); i++ });
  },
  maybeReload(j) { if (j.kind === "quicklook") Runs.loaded = false },
  paint() {
    const s = this.sum; if (!s) return; const show0 = $("#l2bShow0").checked;
    $("#l2bLegend").innerHTML = s.classes.map(c => { const on = c.id ? this.vis.has(c.id) : show0; return `<div class="chip ${on ? "" : "off"}" data-id="${c.id}"><span class="sw" style="background:${c.color}"></span><span class="nm" title="${esc(c.name)}">${c.id === 0 ? "unclassified (id 0)" : esc(c.name)}</span><span class="ct">${fmtN(c.count)}</span></div>` }).join("");
    $$("#l2bLegend .chip").forEach(e => e.onclick = () => { const id = +e.dataset.id; if (id === 0) { $("#l2bShow0").checked = !$("#l2bShow0").checked } else this.vis.has(id) ? this.vis.delete(id) : this.vis.add(id); this.paint() });
    const lut = new Uint32Array(65536); const hex = h => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16));
    s.classes.forEach(c => { const on = c.id ? this.vis.has(c.id) : show0; if (!on) return; const [r, g, b] = hex(c.color); lut[c.id & 0xffff] = (255 << 24) | (b << 16) | (g << 8) | r });
    const ctx = this.cv.getContext("2d"), im = ctx.createImageData(this.cv.width, this.cv.height), u = new Uint32Array(im.data.buffer), ids = this.ids;
    for (let i = 0; i < ids.length; i++) u[i] = lut[ids[i] & 0xffff];
    ctx.putImageData(im, 0, 0); this.view.draw();
  }
};

/* ---------------- RUNS ---------------- */
const Runs = {
  loaded: false, runs: [], cmp: "swipe",
  async load() {
    if (this.loaded) return; this.loaded = true;
    this.runs = await api("/api/runs").catch(() => []);
    const rl = $("#runList");
    rl.innerHTML = this.runs.length ? this.runs.map(r => `<div class="run"><div><span class="badge ${r.state}">${r.state}</span><div class="muted small" style="margin-top:6px">${new Date(r.started * 1000).toLocaleString()}</div></div>
      <div><b>${esc(r.kind)}</b> <span class="muted small">· ${fmtT(r.runtime)} · exit ${r.exit_code ?? "—"} · ${esc(r.out_dir)} · ${r.images.length} images</span><div class="cmd">${esc(r.command)}</div></div>
      <div class="acts"><button class="btn sm ghost" data-log="${r.id}">Log</button><button class="btn sm" data-out="${r.id}">Outputs</button></div></div>`).join("")
      : `<div class="emptybox">No runs yet. Launch one from the <a data-goto="run">Run</a> tab — each run is saved to webapp\\runs\\ with its command, exit code, runtime and log.</div>`;
    $$("[data-log]").forEach(b => b.onclick = async () => { const url = window.STATIC ? "api/runs/" + b.dataset.log + ".log" : `/api/runs/${b.dataset.log}/log`; const t = await (await fetch(url)).text(); go("run"); $("#termTitle").textContent = b.dataset.log + ".log"; $("#term").textContent = t });
    $$("[data-out]").forEach(b => b.onclick = () => this.gallery(this.runs.find(r => r.id === b.dataset.out)));
    const opts = [...this.runs.filter(r => r.images.length).map(r => [r.id, `${r.kind} · ${new Date(r.started * 1000).toLocaleString()}`, r.out_dir]), ...(await api("/api/outputs/dirs").catch(() => [])).map(d => ["dir:" + d.dir, "current " + d.dir, d.dir])];
    this.opts = opts; const o = opts.map(([v, l]) => `<option value="${esc(v)}">${esc(l)}</option>`).join("");
    $("#cmpA").innerHTML = o; $("#cmpB").innerHTML = o; if (opts.length > 1) $("#cmpB").selectedIndex = 1;
    ["#cmpA", "#cmpB"].forEach(s => $(s).onchange = () => this.cmpImgs()); $("#cmpImg").onchange = () => this.compare();
    $$("#cmpMode button").forEach(b => b.onclick = () => { $$("#cmpMode button").forEach(x => x.classList.toggle("on", x === b)); this.cmp = b.dataset.m; this.compare() });
    $("#refreshRuns").onclick = () => { this.loaded = false; this.load() };
    this.cmpImgs();
    const dirs = await api("/api/outputs/dirs").catch(() => []);
    this.gallery(this.runs.find(r => r.images.length) || (dirs[0] && { id: null, out_dir: dirs[0].dir }));
  },
  dirOf(v) { const o = this.opts.find(x => x[0] === v); return o ? o[2] : null },
  async imgs(dir) { const L = await api("/api/layers?dir=" + encodeURIComponent(dir)).catch(() => ({ layers: [] })); return L.layers.filter(l => l.type === "figure") },
  async cmpImgs() {
    const a = this.dirOf($("#cmpA").value), b = this.dirOf($("#cmpB").value); if (!a || !b) { $("#cmpView").innerHTML = '<div class="emptybox">Nothing to compare yet — run the smoke and full pipeline first.</div>'; return }
    const [A, B] = await Promise.all([this.imgs(a), this.imgs(b)]); const nb = new Set(B.map(x => x.file.split("/").pop()));
    const common = A.map(x => x.file.split("/").pop()).filter(n => nb.has(n));
    $("#cmpImg").innerHTML = common.map(n => `<option>${esc(n)}</option>`).join(""); this.compare();
  },
  compare() {
    const a = this.dirOf($("#cmpA").value), b = this.dirOf($("#cmpB").value), n = $("#cmpImg").value, v = $("#cmpView");
    if (!a || !b || !n) { v.innerHTML = '<div class="emptybox">No image name present in both outputs yet.</div>'; return }
    const ua = fileUrl(a + "/" + n), ub = fileUrl(b + "/" + n);
    if (this.cmp === "side") { v.innerHTML = `<div class="side"><div><div class="muted small">${esc(a)}</div><img src="${ua}"></div><div><div class="muted small">${esc(b)}</div><img src="${ub}"></div></div>`; return }
    v.innerHTML = `<div class="swipe"><img src="${ub}" class="base"><div class="top" style="width:50%"><img src="${ua}"></div><div class="handle" style="left:50%"></div><span class="lab" style="left:10px">A · ${esc(a)}</span><span class="lab" style="right:10px">B · ${esc(b)}</span></div>`;
    const sw = $(".swipe", v), top = $(".top", sw), h = $(".handle", sw), base = $(".base", sw), ti = $("img", top);
    const sync = () => { ti.style.width = base.clientWidth + "px" }; base.onload = sync; new ResizeObserver(sync).observe(sw);
    let dr = false; const mv = e => { if (!dr) return; const r = sw.getBoundingClientRect(); const p = Math.min(100, Math.max(0, (e.clientX - r.left) / r.width * 100)); top.style.width = p + "%"; h.style.left = p + "%" };
    h.onmousedown = () => dr = true; window.addEventListener("mouseup", () => dr = false); window.addEventListener("mousemove", mv);
  },
  async gallery(r) {
    if (!r) { $("#gallery").innerHTML = '<div class="emptybox">No outputs yet — run the pipeline.</div>'; return }
    $("#galTitle").textContent = "Outputs · " + r.out_dir; $("#galSub").textContent = r.id ? "from run " + r.id + " (current files in that folder)" : "current files";
    const ims = await this.imgs(r.out_dir);
    $("#gallery").innerHTML = ims.length ? ims.map(i => `<div class="thumb"><img loading="lazy" src="${fileUrl(i.file)}" data-cap="${esc(i.file)}"><div class="cap"><span title="${esc(i.file)}">${esc(i.file.split("/").pop())}</span><a class="btn sm ghost" href="${fileUrl(i.file, 1)}">↓</a></div></div>`).join("") : '<div class="emptybox">No PNGs in this folder yet.</div>';
    $$("#gallery img").forEach(im => im.onclick = () => { $("#lightbox img").src = im.src; $(".lb-cap").textContent = im.dataset.cap; $("#lightbox").classList.add("show") });
  }
};
$("#lightbox").onclick = () => $("#lightbox").classList.remove("show");


/* ---------------- charts (plain SVG) ---------------- */
const NS = "http://www.w3.org/2000/svg";
function svgEl(t, a, p) { const e = document.createElementNS(NS, t); for (const k in a) e.setAttribute(k, a[k]); p && p.appendChild(e); return e }
const nice = v => Math.abs(v) >= 1e5 ? (v / 1e3).toFixed(0) + "k" : Math.abs(v) >= 1e3 ? (v / 1e3).toFixed(v % 1000 ? 1 : 0) + "k" : Math.abs(v) < 0.01 && v ? v.toExponential(1) : +v.toPrecision(3);
function chart(el, o) {
  el.innerHTML = ""; const W = el.clientWidth || 520, H = el.clientHeight || 250;
  if (!o.y || !o.y.length) { el.innerHTML = '<div class="emptybox">No data yet — run the pipeline.</div>'; return }
  const P = { l: 54, r: o.y2 ? 46 : 14, t: 12, b: o.rot ? 78 : 42 }, n = o.y.length;
  const ymax = (o.ymax ?? Math.max(...o.y) * 1.08) || 1, iw = W - P.l - P.r, ih = H - P.t - P.b, bw = iw / n;
  const X = i => P.l + i * bw, Y = v => P.t + ih - v / ymax * ih;
  const s = svgEl("svg", { viewBox: `0 0 ${W} ${H}` }, el); const tip = document.createElement("div"); tip.className = "tip"; el.appendChild(tip);
  for (let k = 0; k <= 4; k++) { const v = ymax * k / 4, y = Y(v); svgEl("line", { x1: P.l, x2: W - P.r, y1: y, y2: y, class: "grid" }, s); svgEl("text", { x: P.l - 7, y: y + 3.5, "text-anchor": "end", class: "ax" }, s).textContent = (o.fy || nice)(v) }
  o.y.forEach((v, i) => {
    const gap = o.hist ? 0.6 : Math.min(10, bw * 0.2); const r = svgEl("rect", { x: X(i) + gap, y: Y(Math.max(0, v)), width: Math.max(0.5, bw - 2 * gap), height: Math.max(0, Y(0) - Y(Math.max(0, v))), fill: typeof o.color === "function" ? o.color(i) : (o.color || "#5B6770"), class: "bar" }, s);
    r.addEventListener("mousemove", e => { const b = el.getBoundingClientRect(); tip.innerHTML = o.tip ? o.tip(i) : `${o.x[i]}: ${nice(v)}`; tip.style.opacity = 1; tip.style.left = Math.min(e.clientX - b.left + 12, W - tip.offsetWidth - 4) + "px"; tip.style.top = (e.clientY - b.top - 30) + "px" });
    r.addEventListener("mouseleave", () => tip.style.opacity = 0);
  });
  if (o.hist) { const e = o.edges, ticks = 6; for (let k = 0; k <= ticks; k++) { const i = Math.round(k * n / ticks), x = P.l + i * bw; svgEl("line", { x1: x, x2: x, y1: Y(0), y2: Y(0) + 4, stroke: "#B7B2A6" }, s); svgEl("text", { x, y: Y(0) + 16, "text-anchor": "middle", class: "ax" }, s).textContent = nice(e[i]) } }
  else o.x.forEach((l, i) => { const x = X(i) + bw / 2, t = svgEl("text", { x, y: Y(0) + 15, "text-anchor": o.rot ? "end" : "middle", class: "ax" }, s); t.textContent = l; if (o.rot) t.setAttribute("transform", `rotate(-38 ${x} ${Y(0) + 12})`) });
  if (o.mark != null) { const i = o.edges.findIndex(v => v > o.mark) - 1; if (i >= 0) { const x = P.l + (i + (o.mark - o.edges[i]) / (o.edges[i + 1] - o.edges[i])) * bw; svgEl("line", { x1: x, x2: x, y1: P.t, y2: Y(0), stroke: "#CC785C", "stroke-dasharray": "4 3" }, s); svgEl("text", { x: x + 5, y: P.t + 10, class: "ax", fill: "#B4654B" }, s).textContent = o.markLabel || "" } }
  if (o.y2) { const pts = o.y2.map((v, i) => `${X(i) + bw / 2},${P.t + ih - v * ih}`).join(" "); svgEl("polyline", { points: pts, fill: "none", stroke: "#CC785C", "stroke-width": 1.5 }, s); o.y2.forEach((v, i) => svgEl("circle", { cx: X(i) + bw / 2, cy: P.t + ih - v * ih, r: 3, fill: "#FAF9F5", stroke: "#CC785C", "stroke-width": 1.5 }, s)); [0, .5, 1].forEach(v => svgEl("text", { x: W - P.r + 6, y: P.t + ih - v * ih + 3.5, class: "ax" }, s).textContent = (v * 100) + "%") }
  o.xt && (svgEl("text", { x: P.l + iw / 2, y: H - 4, "text-anchor": "middle", class: "axt" }, s).textContent = o.xt);
  if (o.yt) { const t = svgEl("text", { x: 12, y: P.t + ih / 2, "text-anchor": "middle", class: "axt", transform: `rotate(-90 12 ${P.t + ih / 2})` }, s); t.textContent = o.yt }
}

/* ---------------- PIPELINE ---------------- */
const Pipeline = {
  loaded: false, sel: "mf", F: null,
  async init() {
    if (!this.bound) { this.bound = true; window.addEventListener("resize", () => { if ($("#page-pipeline").classList.contains("active") && this.F) this.charts() }) }
    if (this.loaded) return; this.loaded = true;
    const [F, mf, sam, st] = await Promise.all([api("/api/pipeline/facts?dir=outputs").catch(e => ({ error: e.message })), api("/api/stats/mf?dir=outputs").catch(() => ({ minerals: [] })), api("/api/stats/sam?dir=outputs").catch(() => ({ labels: [] })), api("/api/status").catch(() => ({ outputs: [] }))]);
    this.F = F; this.mf = mf; this.sam = sam; this.st = st;
    $("#pipeSrc").textContent = F.log ? "numbers from " + F.log : "no run log found yet";
    const d = F.done || [], t = F.tests;
    $("#kpis").innerHTML = [[d[0] != null ? d[0] + " s" : "—", "full-scene runtime"], [d[1] != null ? fmtN(d[1]) + " MB" : "—", "peak memory"], [t ? t.passed : "—", t ? `tests passed${t.failed ? ", " + t.failed + " failed" : ""}` : "tests (run them)"],
      [fmtN(F.prepared && F.prepared[0]), "spectra mapped"], [fmtN(F.fit && F.fit[0]), "fit sample"], [F.chunks?.length || "—", "row chunks / pass"]].map(([b, s]) => `<div class="kpi"><b>${b}</b><span>${s}</span></div>`).join("");
    this.diagram(); this.show(this.sel); this.charts();
  },
  steps() {
    const F = this.F || {}, sc = F.scene || [], bx = F.boxes || [], p1 = F.pass1 || [], pe = F.pca_explained || [], ev = F.mnf_eigenvalues || [];
    const H = sc[0] || 1280, Wd = sc[1] || 1242, B = sc[2] || 285, gb = sc[3] || 225, cr = sc[5] || 128, prep = F.prepared?.[0];
    const cum = pe.reduce((a, v) => (a.push((a.at(-1) || 0) + v), a), []);
    const pct = v => v == null ? "—" : (100 * v).toFixed(1) + "%";
    const mfTop = (this.mf?.minerals || []).slice().sort((a, b) => b.p99 - a.p99).slice(0, 3).map(m => `${m.name} (${m.p99.toFixed(3)})`).join(", ");
    const sa = this.sam?.angles; const samTop = (this.sam?.labels || []).filter(l => l.label >= 0).sort((a, b) => b.count - a.count).slice(0, 3).map(l => `${l.name} ${pct(l.frac)}`).join(", ");
    const t2 = F.chunks?.length ? F.chunks.map((c, i) => c.elapsed - (i ? F.chunks[i - 1].elapsed : 0)) : [];
    return [
      { id: "l2a", t: "L2A reflectance", s: `${H}×${Wd}×${B} · 1.85 GB`, h: "EMIT L2A surface reflectance", p: "The granule is opened lazily with h5py through io_cube.ReflectanceReader. Only one block of rows is ever in memory; fill values ≤ −999 become NaN and wavelengths are converted from µm to nm.", kv: [["granule", "2605105_013 (2026-02-20)"], ["shape (rows, cols, bands)", `${H} × ${Wd} × ${B}`], ["file size", "1.85 GB"], ["rows per chunk", cr]] },
      { id: "bands", t: "Good bands", s: `${gb} of ${B} kept`, h: "Atmospheric good-band mask", p: "Bands inside the strong water-vapour absorptions and beyond the SWIR-2 edge are dropped before anything else, so noisy channels never enter a covariance.", eq: "keep λ ∈ (450, 1320) ∪ (1460, 1780) ∪ (1970, 2450) nm", kv: [["bands kept", `${gb} / ${B}`], ["bands dropped", B - gb]] },
      { id: "masks", t: "Masks", s: `${bx[0] ?? 22} boxes · ${fmtN(bx[1])} px`, h: "Exclude boxes and quality mask", p: "Manual half-open [row0, row1) × [col0, col1) boxes remove known artefacts. The L2A MASK from the same orbit matches the cube's shape but not its footprint, so it was checked against lat/lon and rejected rather than masking the wrong pixels.", kv: [["exclude boxes", bx[0] ?? "—"], ["excluded pixels", fmtN(bx[1])], ["share of scene", bx[1] ? pct(bx[1] / (H * Wd)) : "—"], ["L2A quality mask", F.qm_offset ? "rejected" : "not used"], ["lat/lon offset", F.qm_offset ? F.qm_offset[0] + "°" : "—"], ["flags it would use", "cloud, cirrus, water, spacecraft"]] },
      { id: "pass1", t: "Pass 1 · sample", s: `${F.pass1_chunks || 10} chunks of ${cr} rows`, h: "Streaming pass 1: thresholds and reservoirs", p: "Each chunk is read, masked and reduced to SWIR brightness so one scene-wide shadow threshold can be set. Reservoir sampling keeps a bounded random sample of valid spectra and of along-track difference pairs for the noise model — the full cube is never materialised.", eq: "noise(r, c) = x(r+1, c) − x(r, c),  both pixels valid", kv: [["chunks", F.pass1_chunks || "—"], ["finite pixels", fmtN(p1[0])], ["SWIR shadow threshold (8th pct)", p1[1] ?? "—"], ["signal reservoir", p1[2] ? `${fmtN(p1[2])} of ${fmtN(p1[3])}` : "—"], ["noise pairs", p1[4] ? `${fmtN(p1[4])} of ${fmtN(p1[5])}` : "—"]] },
      { id: "prep", t: "prepare_spectra", s: `${fmtN(prep)} px survive`, h: "Spectral preparation", p: "Pixels in shadow or with vegetation are dropped, the rest are Savitzky–Golay smoothed and divided by their upper convex hull so that absorption depth, not albedo, drives the mapping.", eq: "x̂(λ) = SG₁₁,₂[x](λ) / hull⁺(λ)", kv: [["shadow cut", `SWIR mean < ${p1[1] ?? "8th pct"}`], ["vegetation cut", "NDVI > 0.25"], ["smoothing", "Savitzky–Golay, window 11, order 2"], ["continuum", "upper convex hull"], ["prepared spectra", fmtN(prep)], ["of finite pixels", prep && p1[0] ? pct(prep / p1[0]) : "—"]] },
      { id: "pca", fit: 1, t: "PCA", s: `${pe.length || 8} components`, h: "Principal component analysis", p: "Fitted on the reservoir sample, then every prepared pixel is projected in pass 2. The RGB product is a 2–98 % stretch of PC1–3.", eq: "scores = (x − μ) · U₁…ₖ,   Σ = U Λ Uᵀ", kv: [["components", pe.length || "—"], ["PC1 / PC2 / PC3", pe.slice(0, 3).map(pct).join(" / ") || "—"], ["cumulative (all)", pct(cum.at(-1))]] },
      { id: "mnf", fit: 1, t: "MNF", s: `λ₁ = ${ev[0] ? ev[0].toFixed(1) : "—"}`, h: "Minimum noise fraction (Green et al.)", p: "Noise covariance comes from one-pixel along-track differences; the generalized symmetric eigenproblem orders components by signal-to-noise rather than variance.", eq: "Σ_signal v = λ Σ_noise v   (scipy.linalg.eigh, λ descending)", kv: [["components", ev.length || "—"], ["λ₁ / λ₂ / λ₃", ev.slice(0, 3).map(v => v.toFixed(2)).join(" / ") || "—"], ["noise pairs", fmtN(p1[4])]] },
      { id: "mf", fit: 1, t: "Matched filter", s: `${F.targets?.[0] ?? 13} USGS targets`, h: "Matched filter", p: "Background mean and covariance come from the 80,000-spectrum sample. Ledoit–Wolf shrinkage is used because N > B; the system is solved with np.linalg.solve and scores are left unclipped.", eq: "score = (t − m)ᵀ Σ⁻¹ (x − m) / (t − m)ᵀ Σ⁻¹ (t − m)", kv: [["targets", F.targets?.[0] ?? "—"], ["covariance", "Ledoit–Wolf (N > B)"], ["background sample", fmtN(F.fit?.[0])], ["highest 99th-pct scores", mfTop || "—"]] },
      { id: "sam", fit: 1, t: "SAM", s: "threshold 0.12 rad", h: "Spectral angle mapper", p: "Each prepared spectrum takes the label of the library spectrum with the smallest angle; pixels whose best angle exceeds 0.12 rad are labelled −1.", eq: "θ = arccos( x · t / ‖x‖ ‖t‖ ),   label = −1 if θ > 0.12", kv: [["threshold", "0.12 rad"], ["median best angle", sa ? sa.median.toFixed(3) + " rad" : "—"], ["pixels ≤ 0.12 rad", sa ? pct(sa.frac_le_012) : "—"], ["largest classes", samTop || "—"]] },
      { id: "win", fit: 1, t: "Feature windows", s: `${F.windows?.length || 5} window PCAs`, h: "Diagnostic-window PCA", p: "PCA restricted to absorption windows isolates subtle features — Fe³⁺ in the VNIR, Mg-OH and carbonate in the SWIR. At most five components per window.", kv: (F.windows?.length ? F.windows : [{ name: "hematite" }, { name: "chlorite" }, { name: "carbonate" }, { name: "alteration" }, { name: "serpentine" }]).map(w => [w.name + " " + ({ hematite: "550–950", chlorite: "2180–2265", carbonate: "2280–2380", alteration: "2100–2380", serpentine: "2280–2340" }[w.name] || "") + " nm", w.bands ? w.bands + " bands" : "—"]) },
      { id: "pass2", t: "Pass 2 · memmaps", s: "float32 (H, W, k)", h: "Streaming pass 2: transform every pixel", p: "The cube is streamed again; every prepared pixel is projected/scored and written straight into float32 .npy memmaps, so memory stays bounded by one chunk.", kv: [["prepared spectra", fmtN(prep)], ["chunks", F.chunks?.length || "—"], ["mean time / chunk", t2.length ? (t2.reduce((a, b) => a + b, 0) / t2.length).toFixed(0) + " s" : "—"], ["total runtime", F.done ? F.done[0] + " s" : "—"], ["peak memory", F.done ? fmtN(F.done[1]) + " MB" : "—"]] },
      { id: "png", t: "Products", s: `${(this.st?.outputs || []).find(d => d.dir === "outputs")?.png ?? "—"} PNGs`, h: "Rendered products", p: "PNG figures are rendered from the memmaps with src.plotting. Explore them interactively in Map and Scores.", kv: [["PCA / MNF", "pca_rgb.png, mnf_rgb.png"], ["matched filter", "mf_{mineral}.png, mf_best_labels.png"], ["SAM", "sam_labels.png, sam_angles.png"], ["windows", "window_{name}_pca_rgb.png"]] },
    ];
  },
  diagram() {
    const S = this.steps(), main = S.filter(s => !s.fit && s.id !== "pass2" && s.id !== "png"), fits = S.filter(s => s.fit);
    const el = $("#diagram"); el.innerHTML = ""; const svg = svgEl("svg", { viewBox: "0 0 1240 360" }, el);
    const defs = svgEl("defs", {}, svg), mk = svgEl("marker", { id: "ar", viewBox: "0 0 8 8", refX: 7, refY: 4, markerWidth: 7, markerHeight: 7, orient: "auto" }, defs); svgEl("path", { d: "M0,0 L8,4 L0,8 z", fill: "#B7B2A6" }, mk);
    const pos = {}; const MW = 124, MH = 62, MY = 160;
    main.forEach((s, i) => pos[s.id] = { x: 6 + i * 144, y: MY, w: MW, h: MH });
    fits.forEach((s, i) => pos[s.id] = { x: 760, y: 34 + i * 64, w: 168, h: 48 });
    pos.pass2 = { x: 966, y: MY, w: 128, h: MH }; pos.png = { x: 1110, y: MY, w: 124, h: MH };
    svgEl("text", { x: 760, y: 20, class: "lane" }, svg).textContent = "FITS ON THE PASS-1 SAMPLE";
    svgEl("text", { x: 6, y: MY - 14, class: "lane" }, svg).textContent = "STREAMED IN ROW CHUNKS";
    const edge = (a, b) => { const A = pos[a], Bp = pos[b], x1 = A.x + A.w, y1 = A.y + A.h / 2, x2 = Bp.x - 3, y2 = Bp.y + Bp.h / 2, mx = (x1 + x2) / 2; svgEl("path", { d: `M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}`, class: "edge", "marker-end": "url(#ar)" }, svg) };
    main.slice(1).forEach((s, i) => edge(main[i].id, s.id)); fits.forEach(f => { edge("prep", f.id); edge(f.id, "pass2") }); edge("pass2", "png");
    S.forEach(s => { const p = pos[s.id], g = svgEl("g", { class: "node", "data-id": s.id }, svg); svgEl("rect", { x: p.x, y: p.y, width: p.w, height: p.h, rx: 8 }, g); const c = p.h > 50; svgEl("text", { x: p.x + 12, y: p.y + (c ? 26 : 21), class: "t" }, g).textContent = s.t; svgEl("text", { x: p.x + 12, y: p.y + (c ? 45 : 37), class: "s" }, g).textContent = s.s; g.onclick = () => this.show(s.id) });
  },
  show(id) {
    this.sel = id; const s = this.steps().find(x => x.id === id); $$("#diagram .node").forEach(g => g.classList.toggle("on", g.dataset.id === id));
    $("#stepDetail").innerHTML = `<div><h3>${esc(s.h)}</h3><p>${esc(s.p)}</p>${s.eq ? `<div class="eq">${esc(s.eq)}</div>` : ""}</div><div><table class="kv">${s.kv.map(([k, v]) => `<tr><td>${esc(k)}</td><td>${esc(v)}</td></tr>`).join("")}</table></div>`;
  },
  charts() {
    const F = this.F || {}, pe = F.pca_explained || [], ev = F.mnf_eigenvalues || [];
    const cum = pe.reduce((a, v) => (a.push((a.at(-1) || 0) + v), a), []);
    chart($("#chPca"), { x: pe.map((_, i) => "PC" + (i + 1)), y: pe, y2: cum, ymax: Math.max(...pe, 0) * 1.15, fy: v => (v * 100).toFixed(0) + "%", color: "#5B6770", yt: "variance ratio", tip: i => `PC${i + 1}: ${(pe[i] * 100).toFixed(2)}% · cumulative ${(cum[i] * 100).toFixed(1)}%` });
    chart($("#chMnf"), { x: ev.map((_, i) => "C" + (i + 1)), y: ev, color: "#7A7D5C", yt: "eigenvalue (SNR)", tip: i => `MNF ${i + 1}: λ = ${ev[i].toFixed(3)}` });
    const ms = this.mf?.minerals || []; const sel = $("#mfSel");
    if (!sel.options.length) { sel.innerHTML = ms.map(m => `<option>${esc(m.name)}</option>`).join(""); sel.onchange = () => this.charts() }
    const m = ms.find(x => x.name === sel.value) || ms[0];
    if (m) { chart($("#chMf"), { hist: 1, edges: m.edges, y: m.counts, color: "#CC785C", xt: "matched-filter score", yt: "pixels (sampled)", mark: 0, markLabel: "0", tip: i => `${nice(m.edges[i])} … ${nice(m.edges[i + 1])}: ${fmtN(m.counts[i])} px` }); $("#mfInfo").textContent = `median ${m.p50.toFixed(4)} · 99th pct ${m.p99.toFixed(3)} · ${(m.frac_gt_0_1 * 100).toFixed(2)}% of pixels score > 0.1 · n = ${fmtN(m.n)} sampled` } else chart($("#chMf"), { y: [] });
    const L = (this.sam?.labels || []).filter(l => l.label >= 0).sort((a, b) => b.count - a.count);
    chart($("#chSam"), { x: L.map(l => l.name), y: L.map(l => l.frac), rot: 1, fy: v => (v * 100).toFixed(0) + "%", color: i => L[i].label < 0 ? "#D6D2C7" : "#6E8B8E", tip: i => `${esc(L[i].name)}: ${(L[i].frac * 100).toFixed(2)}% · ${fmtN(L[i].count)} px` });
    const a = this.sam?.angles; if (a) { chart($("#chAng"), { hist: 1, edges: a.edges, y: a.counts, color: "#8C5E58", mark: 0.12, markLabel: "0.12 rad", xt: "best spectral angle (rad)", yt: "pixels (sampled)", tip: i => `${a.edges[i].toFixed(3)} … ${a.edges[i + 1].toFixed(3)} rad: ${fmtN(a.counts[i])} px` }); $("#angInfo").textContent = `median ${a.median.toFixed(3)} rad · ${(a.frac_le_012 * 100).toFixed(1)}% within 0.12 rad` } else chart($("#chAng"), { y: [] });
    const C = F.chunks || []; const dt = C.map((c, i) => c.elapsed - (i ? C[i - 1].elapsed : 0));
    chart($("#chChunks"), { x: C.map(c => `${c.r0}–${c.r1}`), y: C.map(c => c.prepared), rot: 1, color: "#A3866A", yt: "prepared pixels", tip: i => `rows ${C[i].r0}–${C[i].r1}: ${fmtN(C[i].prepared)} px · ${dt[i]} s (t = ${C[i].elapsed} s)` });
  }
};

/* ---------------- SCORES ---------------- */
const Scores = {
  loaded: false, dir: "outputs", pc: 0, mf: null,
  url(path, o) { const s = +$("#scStretch").value; const q = new URLSearchParams({ path, lo: s, hi: 100 - s, cmap: o.cmap }); if (o.component != null) q.set("component", o.component); if (!window.STATIC) return "/api/render?" + q; const info = renderUrl(q); return info.url; },
  async init() {
    if (!this.bound) { this.bound = true; $("#scStretch").onchange = () => this.draw(true); $("#scStretch").oninput = () => $("#scStretchLbl").textContent = `${$("#scStretch").value}–${100 - $("#scStretch").value}%`; $("#scCmap").onchange = () => this.draw(true); $("#mfBigSel").onchange = () => { this.mf = $("#mfBigSel").value; this.draw() } }
    if (this.loaded) return; this.loaded = true;
    const dirs = await api("/api/outputs/dirs").catch(() => []);
    $("#scDir").innerHTML = dirs.map(d => `<button data-d="${d.dir}" class="${d.dir === this.dir ? "on" : ""}">${d.dir}</button>`).join("");
    $$("#scDir button").forEach(b => b.onclick = () => { this.dir = b.dataset.d; this.loaded = false; this.init() });
    const L = await api("/api/layers?dir=" + encodeURIComponent(this.dir)).catch(() => ({ layers: [] }));
    this.pca = L.layers.find(l => /pca_scores/.test(l.file)); this.mfs = L.layers.filter(l => /\/mf_(?!best).*\.npy$/.test(l.file));
    const k = this.pca ? this.pca.shape[2] : 0;
    $("#pcBtns").innerHTML = [...Array(k).keys()].map(i => `<button data-i="${i}" class="${i === this.pc ? "on" : ""}">PC${i + 1}</button>`).join("");
    $$("#pcBtns button").forEach(b => b.onclick = () => { this.pc = +b.dataset.i; this.draw() });
    if (!this.mfs.find(m => m.label === this.mf)) this.mf = this.mfs[0]?.label;
    $("#mfBigSel").innerHTML = this.mfs.map(m => `<option ${m.label === this.mf ? "selected" : ""}>${esc(m.label)}</option>`).join("");
    this.draw(true);
  },
  draw(all) {
    const cm = $("#scCmap").value;
    if (this.pca) { $("#pcBig").innerHTML = `<img src="${this.url(this.pca.file, { cmap: "gray", component: this.pc })}" data-fb="files/outputs/pca_rgb.png" onerror="if(this.dataset.fb){this.onerror=null;this.src=this.dataset.fb}">`; $("#pcCap").textContent = `PC${this.pc + 1} of ${this.pca.shape[2]} · ${this.pca.file}`; $$("#pcBtns button").forEach(b => b.classList.toggle("on", +b.dataset.i === this.pc)) } else $("#pcBig").innerHTML = '<span class="muted">No pca_scores.npy in this folder yet.</span>';
    const m = this.mfs.find(x => x.label === this.mf);
    if (m) { const fb = figureFallback(m.file); $("#mfBig").innerHTML = `<img src="${this.url(m.file, { cmap: cm })}" ${fb ? `data-fb="${fb}" onerror="if(this.dataset.fb){this.onerror=null;this.src=this.dataset.fb}"` : ""}>`; $("#mfCap").textContent = `${m.label} · ${m.file}`; $("#mfBigSel").value = m.label } else $("#mfBig").innerHTML = '<span class="muted">No matched-filter maps yet.</span>';
    if (all) {
      $("#pcThumbs").innerHTML = this.pca ? [...Array(this.pca.shape[2]).keys()].map(i => `<div class="th" data-i="${i}"><img loading="lazy" src="${this.url(this.pca.file, { cmap: "gray", component: i })}" data-fb="files/outputs/pca_rgb.png" onerror="if(this.dataset.fb){this.onerror=null;this.src=this.dataset.fb}"><span>PC${i + 1}</span></div>`).join("") : '<div class="emptybox">No outputs yet — run the pipeline.</div>';
      $("#mfThumbs").innerHTML = this.mfs.length ? this.mfs.map(x => { const fb = figureFallback(x.file); return `<div class="th" data-n="${esc(x.label)}"><img loading="lazy" src="${this.url(x.file, { cmap: cm })}" ${fb ? `data-fb="${fb}" onerror="if(this.dataset.fb){this.onerror=null;this.src=this.dataset.fb}"` : ""}><span>${esc(x.label)}</span></div>`; }).join("") : '<div class="emptybox">No outputs yet — run the pipeline.</div>';
      $$("#pcThumbs .th").forEach(t => t.onclick = () => { this.pc = +t.dataset.i; this.draw(); window.scrollTo({ top: $("#pcBig").getBoundingClientRect().top + scrollY - 120, behavior: "smooth" }) });
      $$("#mfThumbs .th").forEach(t => t.onclick = () => { this.mf = t.dataset.n; this.draw(); window.scrollTo({ top: $("#mfBig").getBoundingClientRect().top + scrollY - 120, behavior: "smooth" }) });
    }
    $$("#pcThumbs .th").forEach(t => t.classList.toggle("on", +t.dataset.i === this.pc)); $$("#mfThumbs .th").forEach(t => t.classList.toggle("on", t.dataset.n === this.mf));
  }
};

/* ---------------- boot ---------------- */
async function boot() {
  await detectStatic();
  if (window.STATIC) {
    applyStaticUI();
    try {
      const t = await (await fetch("api/runs/full-013-cli.log")).text();
      const term = $("#term");
      if (term && (!term.textContent || term.textContent.includes("ready"))) {
        $("#termTitle").textContent = "full 013 scene · completed run";
        term.textContent = t;
        const facts = await api("/api/pipeline/facts?dir=outputs").catch(() => null);
        if (facts && facts.done) {
          $("#elapsed").textContent = fmtT(facts.done[0]);
          $("#phase").textContent = `Completed in ${facts.done[0]} s · peak ${fmtN(facts.done[1])} MB · read-only showcase`;
          $("#progBar").style.width = "100%";
        }
      }
    } catch (e) { }
  }
  Run.init();
  loadStatus();
  if (!window.STATIC) {
    setInterval(loadStatus, 15000);
    api("/api/jobs/current").then(j => j && Run.attach(j)).catch(() => { });
  }
  const def = window.STATIC ? "pipeline" : "run";
  go((location.hash || ("#" + def)).slice(1) in pages ? (location.hash || ("#" + def)).slice(1) : def);
}
boot();
