// Static browser app: FNO via onnxruntime-web (wasm) + conjugate-gradient reference solver.
// Every fetch uses a relative URL, so the page works under a GitHub Pages subpath.
import { solveDarcy } from "./solver.js";

const ORT_VERSION = "1.20.1";  // must match the <script> tag in index.html
const N = 128, LO = 0.1, HI = 1.0;
const UNTRAINED = "untrained (random weights)";
const STORE_KEY = "fno-darcy-model";

/* global ort */
ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
ort.env.wasm.numThreads = 1;

// Colormap stops sampled from matplotlib's viridis / magma / inferno (same maps as src/infer.py).
const CMAPS = {
  viridis: ["#440154","#482878","#3e4a89","#31688e","#26828e","#1f9e89","#35b779","#6ece58","#b5de2b","#fde725"],
  magma:   ["#000004","#180f3d","#440f76","#721f81","#9e2f7f","#cd4071","#f1605d","#fd9668","#feca8d","#fcfdbf"],
  inferno: ["#000004","#1b0c41","#4a0c6b","#781c6d","#a52c60","#cf4446","#ed6925","#fb9b06","#f7d13d","#fcffa4"],
};
const PANELS = [
  { key: "a",    title: "permeability a(x)", note: "input · drag to paint", cmap: "viridis" },
  { key: "pred", title: "FNO u(x)",          note: "shared scale",          cmap: "magma" },
  { key: "u",    title: "solver u(x)",       note: "shared scale",          cmap: "magma" },
  { key: "err",  title: "|difference|",      note: "|FNO − solver|",        cmap: "inferno" },
];
const rgb = h => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16));
const LUT = Object.fromEntries(Object.entries(CMAPS).map(([k, stops]) => {
  const s = stops.map(rgb), lut = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = i / 255 * (s.length - 1), j = Math.min(Math.floor(t), s.length - 2), f = t - j;
    for (let c = 0; c < 3; c++) lut[i * 3 + c] = s[j][c] + (s[j + 1][c] - s[j][c]) * f;
  }
  return [k, lut];
}));

const $ = id => document.getElementById(id);
const fmt = v => v === 0 ? "0" : Math.abs(v) >= 1e-2 && Math.abs(v) < 1e3 ? v.toFixed(3) : v.toExponential(2);
const pct = v => (100 * v).toFixed(2) + " %";
const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const el = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html) e.innerHTML = html; return e; };
const relL2 = (x, ref) => {
  let d = 0, s = 0;
  for (let i = 0; i < ref.length; i++) { d += (x[i] - ref[i]) ** 2; s += ref[i] ** 2; }
  return Math.sqrt(d / s);
};
const store = {
  get: k => { try { return localStorage.getItem(k); } catch { return null; } },
  set: (k, v) => { try { localStorage.setItem(k, v); } catch { /* storage unavailable */ } },
};
async function fetchOk(url, as) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url} ${r.status}`);
  return r[as]();
}

// Fields are flat Float32Arrays, row-major: index r * N + c, row 0 at the top (y = 1).
const fields = { a: new Float32Array(N * N).fill(LO), u: null, pred: null, err: null };
const canvases = {};
for (const p of PANELS) {
  const fig = el("figure");
  fig.append(el("figcaption", "", `<span>${p.title}</span><small>${p.note}</small>`));
  const map = el("canvas", "map"), bar = el("canvas", "bar"), range = el("div", "range"), cross = el("div", "cross");
  map.width = map.height = N; bar.width = 256; bar.height = 1;
  fig.append(map, bar, range, cross);
  $("panels").append(fig);
  canvases[p.key] = { map, range, cross, cmap: p.cmap };
  const bctx = bar.getContext("2d"), img = bctx.createImageData(256, 1), lut = LUT[p.cmap];
  for (let i = 0; i < 256; i++) img.data.set([lut[i*3], lut[i*3+1], lut[i*3+2], 255], i * 4);
  bctx.putImageData(img, 0, 0);
}

function extent(...fs) {
  let lo = Infinity, hi = -Infinity;
  for (const f of fs) for (const v of f) { if (v < lo) lo = v; if (v > hi) hi = v; }
  return [lo, hi];
}

function draw(key, field, lo, hi) {
  const { map, range, cmap } = canvases[key], lut = LUT[cmap], span = hi - lo || 1;
  const ctx = map.getContext("2d"), img = ctx.createImageData(N, N);
  for (let i = 0; i < field.length; i++) {
    const k = Math.round(Math.min(1, Math.max(0, (field[i] - lo) / span)) * 255) * 3;
    img.data[i*4] = lut[k]; img.data[i*4+1] = lut[k+1]; img.data[i*4+2] = lut[k+2]; img.data[i*4+3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  range.innerHTML = `<span>${fmt(lo)}</span><span>${fmt(hi)}</span>`;
}

function clearPanel(k) {
  fields[k] = null;
  canvases[k].map.getContext("2d").clearRect(0, 0, N, N);
  canvases[k].range.innerHTML = "";
}

const drawA = () => draw("a", fields.a, LO, HI);
function setStatus(text, isError = false) {
  $("status").textContent = text;
  $("status").className = isError ? "error" : "";
}
function showError(text) { $("banner").textContent = text; $("banner").classList.add("on"); }
function hideError() { $("banner").classList.remove("on"); }
// Called whenever a(x) changes: outputs no longer belong to the shown input.
function changed() {
  drawA();
  if ($("auto").checked) run();
  else if (fields.pred) $("stale").textContent = "a(x) changed: press Solve to update";
}

// ---- models (models.json manifest; falls back to the single model.json/model.onnx) ----------
let models = [], current = null;
const sessions = new Map();  // id -> Promise<{ session, test }>
const ready = new Set();     // ids whose session finished loading

async function loadManifest() {
  try {
    return await fetchOk("./models.json", "json");
  } catch {
    const m = await fetchOk("./model.json", "json");
    return { default: "default", models: [{ id: "default", label: m.trained ? "trained FNO" : UNTRAINED, precision: "fp32",
      file: "model.onnx", golden: "golden.json", tol: 1e-4, trained: m.trained, params: m.params, file_mb: null, test_relL2: null }] };
  }
}

const modelName = m => m.trained ? `${m.label}${m.precision === "fp32" ? "" : " · " + m.precision}` : UNTRAINED;

function renderModels() {
  $("models").replaceChildren(...models.map(m => {
    const b = el("button", "model");
    b.setAttribute("role", "radio");
    b.setAttribute("aria-checked", m === current);
    b.dataset.id = m.id;
    const err = m.trained && m.test_relL2 != null ? `test ${pct(m.test_relL2)}` : m.trained ? "" : "untrained";
    b.innerHTML = `<span class="dot"></span><span class="name">${esc(m.trained ? m.label : UNTRAINED)}</span>` +
      `<span class="err">${err}</span>` +
      `<span class="meta">${esc(m.precision)}${m.file_mb ? ` · ${m.file_mb.toFixed(1)} MB` : ""}` +
      ` · <span class="state" id="state-${m.id}">${sessions.has(m.id) ? "loaded" : "not loaded"}</span></span>`;
    b.onclick = () => selectModel(m.id);
    return b;
  }));
}

// Download with a progress bar (content-length may be the compressed size, so clamp to 100 %).
async function download(url, onProgress) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url} ${r.status}`);
  const total = +r.headers.get("content-length") || 0;
  if (!r.body || !total) return new Uint8Array(await r.arrayBuffer());
  const reader = r.body.getReader(), chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value); got += value.length;
    onProgress(Math.min(1, got / total));
  }
  const out = new Uint8Array(got);
  let o = 0;
  for (const c of chunks) { out.set(c, o); o += c.length; }
  return out;
}

async function runSession(session, a) {
  const out = await session.run({ a: new ort.Tensor("float32", Float32Array.from(a), [1, 1, N, N]) });
  return out.u.data;
}

// Self-test: the in-browser model vs the PyTorch output saved in the model's golden.json.
async function selfTest(m, session) {
  const g = await fetchOk("./" + m.golden, "json");
  const u = await runSession(session, g.a);
  let maxErr = 0;
  for (let i = 0; i < u.length; i++) maxErr = Math.max(maxErr, Math.abs(u[i] - g.u[i]));
  const rel = relL2(u, g.u);
  return { ok: Number.isFinite(rel) && rel <= m.tol, rel, maxErr, tol: m.tol };
}

function loadModel(m) {
  if (!sessions.has(m.id)) {
    const p = (async () => {
      const bar = $("progress"), fill = bar.firstElementChild, state = () => $(`state-${m.id}`);
      bar.classList.add("on"); fill.style.width = "0";
      try {
        const bytes = await download("./" + m.file, f => {
          fill.style.width = (100 * f).toFixed(0) + "%";
          if (state()) state().textContent = `downloading ${(100 * f).toFixed(0)} %`;
        });
        if (state()) state().textContent = "starting…";
        const session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
        const test = await selfTest(m, session);
        if (state()) state().textContent = "loaded";
        return { session, test };
      } finally {
        bar.classList.remove("on");
      }
    })();
    p.catch(() => sessions.delete(m.id));  // allow a retry after a failed download
    sessions.set(m.id, p);
  }
  return sessions.get(m.id);
}

function showSelfTest(m, t) {
  const badge = $("selftest");
  if (!t.ok) {
    badge.textContent = "self-test FAILED"; badge.className = "badge bad";
    showError(`Self-test failed for ${modelName(m)}: the in-browser FNO disagrees with PyTorch on ${m.golden} ` +
              `(relL2 ${t.rel.toExponential(2)} > ${t.tol}, max error ${t.maxErr.toExponential(2)}). Predictions are not trustworthy.`);
  } else {
    hideError();
    badge.textContent = `self-test passed · relL2 ${t.rel.toExponential(1)} ≤ ${t.tol} · max err ${t.maxErr.toExponential(1)}`;
    badge.className = "badge ok";
  }
}

function showModelBadge(m) {
  const badge = $("badge");
  badge.textContent = m.trained ? `Predictions: ${modelName(m)} · ${(m.params / 1e6).toFixed(2)}M params · in-browser`
                                : `Predictions: ${UNTRAINED}`;
  badge.className = "badge " + (m.trained ? "ok" : "warn");
}

let selecting = 0;
async function selectModel(id) {
  const m = models.find(x => x.id === id);
  if (!m || m === current) return;
  const token = ++selecting;
  current = m;
  store.set(STORE_KEY, id);
  renderModels();
  showModelBadge(m);
  $("selftest").textContent = "self-test pending"; $("selftest").className = "badge";
  $("solve").disabled = true;
  setStatus(`loading ${modelName(m)}…`);
  try {
    const { test } = await loadModel(m);
    ready.add(m.id);
    if (token !== selecting) return;  // the user picked another model meanwhile
    showSelfTest(m, test);
    setStatus("");
    $("solve").disabled = false;
    if (fields.pred || $("auto").checked) run();
  } catch (e) {
    if (token !== selecting) return;
    setStatus("error: " + e.message, true);
    showError(`Could not load ${modelName(m)}: ${e.message}`);
    renderModels();
  }
}

// ---- solve --------------------------------------------------------------------------------
let busy = false, again = false;
async function run() {
  if (!current || !ready.has(current.id)) return;
  if (busy) { again = true; return; }  // rerun once the current solve finishes, with the newest a(x)
  busy = true; again = false; $("stale").textContent = "";
  setStatus("running FNO and solver…");
  await new Promise(r => setTimeout(r, 0));  // let the status paint before the solver blocks
  const m = current;
  try {
    const { session } = await sessions.get(m.id);
    const a = fields.a.slice();
    let t = performance.now();
    const pred = await runSession(session, a);
    const fnoMs = performance.now() - t;
    t = performance.now();
    const cg = solveDarcy(a, N);
    const cgMs = performance.now() - t;

    fields.pred = Float32Array.from(pred);
    const metric = (label, value, note = "", cls = "") =>
      `<div class="metric ${cls}"><span>${label}</span><b>${value}</b><small>${note}</small></div>`;
    let gap;
    if (cg.converged) {
      fields.u = Float32Array.from(cg.u);
      fields.err = fields.pred.map((v, i) => Math.abs(v - fields.u[i]));
      const [lo, hi] = extent(fields.u, fields.pred);
      draw("pred", fields.pred, lo, hi); draw("u", fields.u, lo, hi);
      draw("err", fields.err, 0, extent(fields.err)[1]);
      gap = metric("relative gap", pct(relL2(fields.pred, fields.u)), `max |diff| ${fmt(extent(fields.err)[1])}`, "hero");
    } else {
      const [lo, hi] = extent(fields.pred);
      draw("pred", fields.pred, lo, hi); clearPanel("u"); clearPanel("err");
      gap = metric("relative gap", "n/a", "solver did not converge");
    }
    const ratio = cgMs / fnoMs;
    const speedup = !(fnoMs > 0) ? "" : ratio >= 1 ? `${ratio.toFixed(1)}× faster than the solver` : `${(1 / ratio).toFixed(1)}× slower than the solver`;
    $("metrics").innerHTML = [
      gap,
      metric("FNO (wasm)", fnoMs.toFixed(0) + " ms", speedup),
      metric("CG solver (JS)", cgMs.toFixed(0) + " ms",
             `${cg.converged ? "" : "did not converge · "}${cg.iters} iters · residual ${cg.relres.toExponential(1)}`,
             cg.converged ? "" : "bad"),
      metric("model", esc(modelName(m)), m.trained && m.test_relL2 != null ? `test-set mean ${pct(m.test_relL2)}` : ""),
    ].join("");
    setStatus(cg.converged ? "" : `solver did not converge after ${cg.iters} iterations (residual ${cg.relres.toExponential(2)}); no solver result shown`, !cg.converged);
  } catch (e) {
    setStatus("error: " + e.message, true);
  } finally {
    busy = false;
    if (again || m !== current) run();
  }
}
$("solve").onclick = run;
$("auto").onchange = () => { if ($("auto").checked && $("stale").textContent) run(); };

// ---- undo history -------------------------------------------------------------------------
const history = [];
function snapshot() { history.push(fields.a.slice()); if (history.length > 30) history.shift(); }
$("undo").onclick = () => { if (history.length) { fields.a = history.pop(); changed(); } };

// ---- tabs ---------------------------------------------------------------------------------
document.querySelectorAll(".tabs button").forEach(b => b.onclick = () => {
  document.querySelectorAll(".tabs button").forEach(t => t.setAttribute("aria-selected", t === b));
  document.querySelectorAll(".pane").forEach(p => p.classList.toggle("on", p.id === "pane-" + b.dataset.pane));
});
const showTab = name => document.querySelector(`.tabs button[data-pane="${name}"]`).click();

// ---- test samples (samples.bin: count x [a, u] x N x N float32) ---------------------------
let samples = null, sampleMeta = null, sampleIdx = -1;
function showSample(k) {
  if (!samples) return;
  sampleIdx = (k + sampleMeta.count) % sampleMeta.count;
  snapshot();
  fields.a = samples.slice(2 * sampleIdx * N * N, (2 * sampleIdx + 1) * N * N);
  $("sampleInfo").textContent = `test sample #${sampleMeta.indices[sampleIdx]} (${sampleIdx + 1} of ${sampleMeta.count} bundled)`;
  showTab("sample");
  changed();
  if (!$("auto").checked) run();  // a new sample always gets solved
}
function randomSample() {
  if (!samples) return;
  let k;
  do { k = Math.floor(Math.random() * sampleMeta.count); } while (k === sampleIdx && sampleMeta.count > 1);
  showSample(k);
}
$("newSample").onclick = randomSample;
$("prevSample").onclick = () => showSample(sampleIdx - 1);
$("nextSample").onclick = () => showSample(sampleIdx + 1);

// ---- drawing ------------------------------------------------------------------------------
let brushValue = HI;
function setBrush(v) {
  brushValue = v;
  $("brushHi").setAttribute("aria-pressed", v === HI); $("brushLo").setAttribute("aria-pressed", v === LO);
}
$("brushHi").onclick = () => setBrush(HI);
$("brushLo").onclick = () => setBrush(LO);
$("fillLo").onclick = () => { snapshot(); fields.a.fill(LO); changed(); };
$("fillHi").onclick = () => { snapshot(); fields.a.fill(HI); changed(); };

const aMap = canvases.a.map;
aMap.classList.add("draw");
function cellAt(e, map) {
  const r = map.getBoundingClientRect();
  return [(e.clientY - r.top) / r.height * N, (e.clientX - r.left) / r.width * N];
}
function stamp(rf, cf) {
  const rad = +$("brush").value / 2;
  for (let r = Math.floor(rf - rad); r <= Math.ceil(rf + rad); r++)
    for (let c = Math.floor(cf - rad); c <= Math.ceil(cf + rad); c++)
      if (r >= 0 && r < N && c >= 0 && c < N && (r + 0.5 - rf) ** 2 + (c + 0.5 - cf) ** 2 <= rad * rad + 0.25)
        fields.a[r * N + c] = brushValue;
}
let last = null;
aMap.addEventListener("pointerdown", e => {
  aMap.setPointerCapture(e.pointerId); snapshot(); showTab("draw");
  last = cellAt(e, aMap); stamp(...last); drawA();
});
aMap.addEventListener("pointermove", e => {
  if (!last) return;
  const [r, c] = cellAt(e, aMap), steps = Math.max(1, Math.ceil(Math.hypot(r - last[0], c - last[1])));
  for (let s = 1; s <= steps; s++) stamp(last[0] + (r - last[0]) * s / steps, last[1] + (c - last[1]) * s / steps);
  last = [r, c]; drawA();
});
const endStroke = () => { if (last) { last = null; changed(); } };
aMap.addEventListener("pointerup", endStroke);
aMap.addEventListener("pointercancel", endStroke);

// ---- function input -----------------------------------------------------------------------
const PRESETS = [
  ["circle",            "hypot(x - 0.5, y - 0.5) < 0.3", true],
  ["two blobs",         "hypot(x - 0.3, y - 0.3) < 0.18 || hypot(x - 0.7, y - 0.65) < 0.22", true],
  ["wavy channel",      "abs(y - 0.5 - 0.2 * sin(2 * PI * x)) < 0.1", true],
  ["diagonal half",     "x + y > 1", true],
  ["checkerboard 4×4",  "(floor(4 * x) + floor(4 * y)) % 2 == 0", true],
  ["smooth bump (not two-phase)", "0.1 + 0.9 * exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) / 0.05)", false],
];
PRESETS.forEach(([name], i) => $("preset").append(new Option(name, i)));
function choosePreset(i) { const [, expr, snap] = PRESETS[i]; $("expr").value = expr; $("snap").checked = snap; }
$("preset").onchange = () => { choosePreset($("preset").value); applyFunction(); };
choosePreset(0);

function applyFunction() {
  $("exprErr").textContent = "";
  let f;
  try {
    // Evaluated only in this browser; there is no server.
    f = new Function("x", "y", `with (Math) { return (${$("expr").value}); }`);
  } catch (e) { $("exprErr").textContent = "syntax error: " + e.message; return; }
  const out = new Float32Array(N * N), snap = $("snap").checked;
  try {
    for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) {
      const v = Number(f((c + 0.5) / N, 1 - (r + 0.5) / N));
      if (!Number.isFinite(v)) throw new Error(`not a finite number at x=${fmt((c + 0.5) / N)}, y=${fmt(1 - (r + 0.5) / N)}`);
      out[r * N + c] = snap ? (v > 0.5 ? HI : LO) : Math.min(HI, Math.max(LO, v));
    }
  } catch (e) { $("exprErr").textContent = e.message; return; }
  snapshot(); fields.a = out; changed();
}
$("apply").onclick = applyFunction;
$("expr").addEventListener("keydown", e => { if (e.key === "Enter") applyFunction(); });

// ---- hover readout + linked crosshair -----------------------------------------------------
const hideCross = () => { for (const c of Object.values(canvases)) c.cross.style.display = "none"; };
for (const { map } of Object.values(canvases)) {
  map.addEventListener("pointermove", e => {
    const [rf, cf] = cellAt(e, map), r = Math.floor(rf), c = Math.floor(cf);
    if (r < 0 || r >= N || c < 0 || c >= N) return;
    for (const o of Object.values(canvases)) {
      o.cross.style.display = "block";
      o.cross.style.left = o.map.offsetLeft + (c + 0.5) / N * o.map.offsetWidth + "px";
      o.cross.style.top = o.map.offsetTop + (r + 0.5) / N * o.map.offsetHeight + "px";
    }
    const i = r * N + c, v = k => fields[k] ? fmt(fields[k][i]) : "–";
    $("readout").textContent = `x ${fmt((c + 0.5) / N)}  y ${fmt(1 - (r + 0.5) / N)}   a ${v("a")}   ` +
      `FNO u ${v("pred")}   solver u ${v("u")}   |difference| ${v("err")}`;
  });
  map.addEventListener("pointerleave", () => { $("readout").textContent = ""; hideCross(); });
}

// ---- keyboard -----------------------------------------------------------------------------
document.addEventListener("keydown", e => {
  const typing = ["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement.tagName);
  if ((e.metaKey || e.ctrlKey) && e.key === "z" && !typing) { e.preventDefault(); $("undo").click(); return; }
  if (typing || e.metaKey || e.ctrlKey || e.altKey) return;
  const k = e.key.toLowerCase();
  if (k === "s" && !$("solve").disabled) run();
  else if (k === "n") randomSample();
  else if (e.key === "ArrowLeft") showSample(sampleIdx - 1);
  else if (e.key === "ArrowRight") showSample(sampleIdx + 1);
  else if (k === "m" && current) selectModel(models[(models.indexOf(current) + 1) % models.length].id);
  else return;
  e.preventDefault();
});

// ---- start --------------------------------------------------------------------------------
drawA();
(async () => {
  try {
    setStatus("loading…");
    const [manifest, bin, sm] = await Promise.all([
      loadManifest(), fetchOk("./samples.bin", "arrayBuffer"), fetchOk("./samples.json", "json"),
    ]);
    samples = new Float32Array(bin); sampleMeta = sm;
    models = manifest.models;
    const wanted = new URLSearchParams(location.search).get("model") || store.get(STORE_KEY);
    const first = models.find(m => m.id === wanted) || models.find(m => m.id === manifest.default) || models[0];
    // Show a sample right away; it is solved as soon as the model is ready.
    sampleIdx = -1;
    showSample(0);
    await selectModel(first.id);
  } catch (e) {
    setStatus("error: " + e.message, true);
    showError("Could not start: " + e.message);
  }
})();
