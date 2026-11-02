// Static browser app: FNO via onnxruntime-web (wasm) + conjugate-gradient reference solver.
// Every fetch uses a relative URL, so the page works under a GitHub Pages subpath.
import { solveDarcy } from "./solver.js";

const ORT_VERSION = "1.20.1";  // must match the <script> tag in index.html
const N = 128, LO = 0.1, HI = 1.0;
const UNTRAINED = "untrained (random weights)";
const SELF_TEST_TOL = 1e-4;

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
  { key: "a",    title: "permeability a(x)", note: "input",                cmap: "viridis" },
  { key: "pred", title: "FNO u(x)",          note: "neural operator",      cmap: "magma" },
  { key: "u",    title: "solver u(x)",       note: "CG, same scale as FNO", cmap: "magma" },
  { key: "err",  title: "|difference|",      note: "|FNO − solver|",       cmap: "inferno" },
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
const el = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html) e.innerHTML = html; return e; };
const relL2 = (x, ref) => {
  let d = 0, s = 0;
  for (let i = 0; i < ref.length; i++) { d += (x[i] - ref[i]) ** 2; s += ref[i] ** 2; }
  return Math.sqrt(d / s);
};

// Fields are flat Float32Arrays, row-major: index r * N + c, row 0 at the top (y = 1).
const fields = { a: new Float32Array(N * N).fill(LO), u: null, pred: null, err: null };
const canvases = {};
for (const p of PANELS) {
  const fig = el("figure");
  fig.append(el("figcaption", "", `${p.title} <small>${p.note}</small>`));
  const map = el("canvas", "map"), bar = el("canvas", "bar"), range = el("div", "range");
  map.width = map.height = N; bar.width = 256; bar.height = 1;
  fig.append(map, bar, range);
  $("panels").append(fig);
  canvases[p.key] = { map, range, cmap: p.cmap };
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
// Called whenever a(x) changes: outputs no longer belong to the shown input.
function changed() { drawA(); if (fields.pred) $("stale").textContent = "a(x) changed: press Solve to update"; }

// ---- model --------------------------------------------------------------------------------
let session = null, modelLabel = "";
async function fno(a) {
  const out = await session.run({ a: new ort.Tensor("float32", Float32Array.from(a), [1, 1, N, N]) });
  return out.u.data;
}

async function selfTest() {
  const g = await (await fetch("./golden.json")).json();
  const u = await fno(g.a);
  let maxErr = 0;
  for (let i = 0; i < u.length; i++) maxErr = Math.max(maxErr, Math.abs(u[i] - g.u[i]));
  const rel = relL2(u, g.u), badge = $("selftest");
  if (rel > SELF_TEST_TOL || !Number.isFinite(rel)) {
    badge.textContent = "self-test FAILED"; badge.className = "badge bad";
    showError(`Self-test failed: the in-browser FNO disagrees with PyTorch on golden.json ` +
              `(relL2 ${rel.toExponential(2)} > ${SELF_TEST_TOL}, max error ${maxErr.toExponential(2)}). Predictions are not trustworthy.`);
  } else {
    badge.textContent = `self-test passed · max err ${maxErr.toExponential(1)} · relL2 ${rel.toExponential(1)}`;
    badge.className = "badge ok";
  }
}

// ---- solve --------------------------------------------------------------------------------
let busy = false;
async function run() {
  if (busy || !session) return;
  busy = true; $("solve").disabled = true; $("stale").textContent = "";
  setStatus("running FNO and solver…");
  await new Promise(r => setTimeout(r, 0));  // let the status paint before the solver blocks
  try {
    const a = fields.a.slice();
    let t = performance.now();
    const pred = await fno(a);
    const fnoMs = performance.now() - t;
    t = performance.now();
    const cg = solveDarcy(a, N);
    const cgMs = performance.now() - t;

    fields.pred = Float32Array.from(pred);
    const metrics = [];
    if (cg.converged) {
      fields.u = Float32Array.from(cg.u);
      fields.err = fields.pred.map((v, i) => Math.abs(v - fields.u[i]));
      const [lo, hi] = extent(fields.u, fields.pred);
      draw("pred", fields.pred, lo, hi); draw("u", fields.u, lo, hi);
      draw("err", fields.err, 0, extent(fields.err)[1]);
      metrics.push(["relative gap ‖FNO − solver‖ / ‖solver‖", (100 * relL2(fields.pred, fields.u)).toFixed(2) + " %"],
                   ["max |difference|", fmt(extent(fields.err)[1])]);
    } else {
      const [lo, hi] = extent(fields.pred);
      draw("pred", fields.pred, lo, hi); clearPanel("u"); clearPanel("err");
      metrics.push(["relative gap", "n/a"]);
    }
    metrics.push(
      ["FNO (wasm)", fnoMs.toFixed(0) + " ms"],
      ["CG solver (JS)", cgMs.toFixed(0) + " ms"],
      ["CG", cg.converged ? `${cg.iters} iters · residual ${cg.relres.toExponential(1)}`
                          : `<b class="bad">did not converge</b>`],
      ["model", modelLabel],
    );
    $("metrics").innerHTML = metrics.map(([k, v]) => `<div><span>${k}</span><b>${v}</b></div>`).join("");
    setStatus(cg.converged ? "" : `solver did not converge after ${cg.iters} iterations (residual ${cg.relres.toExponential(2)}); no solver result shown`, !cg.converged);
  } catch (e) {
    setStatus("error: " + e.message, true);
  } finally {
    busy = false; $("solve").disabled = false;
  }
}
$("solve").onclick = run;

// ---- undo history -------------------------------------------------------------------------
const history = [];
function snapshot() { history.push(fields.a.slice()); if (history.length > 30) history.shift(); }
$("undo").onclick = () => { if (history.length) { fields.a = history.pop(); changed(); } };
document.addEventListener("keydown", e => {
  if ((e.metaKey || e.ctrlKey) && e.key === "z" && document.activeElement.tagName !== "INPUT") { e.preventDefault(); $("undo").click(); }
});

// ---- tabs ---------------------------------------------------------------------------------
document.querySelectorAll(".tabs button").forEach(b => b.onclick = () => {
  document.querySelectorAll(".tabs button").forEach(t => t.setAttribute("aria-selected", t === b));
  document.querySelectorAll(".pane").forEach(p => p.classList.toggle("on", p.id === "pane-" + b.dataset.pane));
});
const showTab = name => document.querySelector(`.tabs button[data-pane="${name}"]`).click();

// ---- test samples (samples.bin: count x [a, u] x N x N float32) ---------------------------
let samples = null, sampleMeta = null, sampleIdx = -1;
function newSample() {
  if (!samples) return;
  let k;
  do { k = Math.floor(Math.random() * sampleMeta.count); } while (k === sampleIdx && sampleMeta.count > 1);
  sampleIdx = k;
  snapshot();
  fields.a = samples.slice(2 * k * N * N, (2 * k + 1) * N * N);
  $("sampleInfo").textContent = `test sample #${sampleMeta.indices[k]} (${k + 1} of ${sampleMeta.count} bundled)`;
  changed();
  run();
}
$("newSample").onclick = newSample;

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

// ---- hover readout ------------------------------------------------------------------------
for (const { map } of Object.values(canvases)) {
  map.addEventListener("pointermove", e => {
    const [rf, cf] = cellAt(e, map), r = Math.floor(rf), c = Math.floor(cf);
    if (r < 0 || r >= N || c < 0 || c >= N) return;
    const i = r * N + c, v = k => fields[k] ? fmt(fields[k][i]) : "–";
    $("readout").textContent = `x ${fmt((c + 0.5) / N)}  y ${fmt(1 - (r + 0.5) / N)}   a ${v("a")}   ` +
      `FNO u ${v("pred")}   solver u ${v("u")}   |difference| ${v("err")}`;
  });
  map.addEventListener("pointerleave", () => { $("readout").textContent = ""; });
}

// ---- start --------------------------------------------------------------------------------
drawA();
(async () => {
  try {
    setStatus("loading model…");
    const [meta, bin, sm] = await Promise.all([
      fetch("./model.json").then(r => { if (!r.ok) throw new Error(`model.json ${r.status}`); return r.json(); }),
      fetch("./samples.bin").then(r => { if (!r.ok) throw new Error(`samples.bin ${r.status}`); return r.arrayBuffer(); }),
      fetch("./samples.json").then(r => { if (!r.ok) throw new Error(`samples.json ${r.status}`); return r.json(); }),
    ]);
    samples = new Float32Array(bin); sampleMeta = sm;
    session = await ort.InferenceSession.create("./model.onnx", { executionProviders: ["wasm"] });
    modelLabel = meta.trained ? "trained FNO" : UNTRAINED;
    const badge = $("badge");
    badge.textContent = meta.trained ? `Predictions: trained FNO · ${(meta.params / 1e6).toFixed(2)}M params · in-browser`
                                     : `Predictions: ${UNTRAINED}`;
    badge.className = "badge " + (meta.trained ? "ok" : "warn");
    await selfTest();
    $("solve").disabled = false;
    setStatus("");
    newSample();
  } catch (e) {
    setStatus("error: " + e.message, true);
    showError("Could not start: " + e.message);
  }
})();
