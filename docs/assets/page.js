/* Every number is read from the JSON files. Nothing here is a typed result. */
const VARIANTS = ["V0", "V1", "V2", "V3"];
const VCOLOR = { V0: "#8c8c8c", V1: "#5b8db8", V2: "#d9a441", V3: "#e0681f" };
const VLABEL = {
  V0: "V0 single frame",
  V1: "V1 raw pair",
  V2: "V2 registered residual",
  V3: "V3 residual + star mask",
};
const SCORE = {
  msp: "Max softmax",
  maxlogit: "Max logit",
  energy: "Energy",
  mahalanobis: "Mahalanobis",
  knn: "kNN (k=10)",
};
const SCORE_COLOR = {
  msp: "#8c8c8c",
  maxlogit: "#5b8db8",
  energy: "#6a9a5b",
  mahalanobis: "#c45b5b",
  knn: "#e0681f",
};
const REG = { stars: "plate-solved stars", ecc: "ECC fallback", identity: "identity" };

const state = {
  A: null, B: null, C: null, M: null,
  confIdx: 0, budgetIdx: 0,
  session: "2023_03_31", frame: 0,
  layer: "v3", residual: false, trails: false,
  kind: "brightness",
  heatClass: null, heatScore: "knn",
  playing: false, timer: null, skyGen: 0,
};
const images = new Map();
const pointer = { x: null, y: null };

function $(id) { return document.getElementById(id); }

function el(tag, attrs, text) {
  const node = document.createElement(tag);
  if (attrs) {
    for (const [k, v] of Object.entries(attrs)) {
      if (v == null) continue;
      if (k === "class") node.className = v;
      else node.setAttribute(k, v);
    }
  }
  if (text != null) node.textContent = text;
  return node;
}

function n(x, d = 3) {
  const v = Number(x);
  if (!Number.isFinite(v)) return "n/a";
  return v.toFixed(d);
}

function pm(stat, d = 3) {
  return n(stat.mean, d) + " ± " + n(stat.std, d);
}

function cssVar(name) {
  return getComputedStyle(document.body).getPropertyValue(name).trim();
}

function nearestIdx(values, x) {
  let best = 0, dist = Infinity;
  values.forEach((v, i) => {
    const d = Math.abs(v - x);
    if (d < dist) { dist = d; best = i; }
  });
  return best;
}

function budgetStops() {
  return Object.keys(state.B.closed_set.budget).sort((a, b) => Number(a) - Number(b));
}

function confOf() {
  return state.A.variants.V2.threshold_curve.conf[state.confIdx];
}

function sessionFrames() {
  return state.M.sessions[state.session].frames;
}

function binLabels(edges, kind) {
  const unit = kind === "speed" ? "px/frame" : kind === "brightness" ? "DN" : "px";
  const digits = kind === "brightness" ? 0 : 1;
  const labs = [];
  for (let i = 0; i < edges.length - 1; i++) {
    const a = edges[i], b = edges[i + 1];
    const fa = a == null ? null : Number(a).toFixed(digits);
    const fb = b == null ? null : Number(b).toFixed(digits);
    if (a == null) labs.push("under " + fb + " " + unit);
    else if (b == null) labs.push("over " + fa + " " + unit);
    else labs.push(fa + " to " + fb + " " + unit);
  }
  return labs;
}

function gapSentence(kind) {
  const labels = binLabels(state.A.bins[kind].edges, kind);
  let bi = 0, best = -Infinity;
  state.A.variants.V2.breakdowns[kind].forEach((b, i) => {
    const g = b.mean - state.A.variants.V1.breakdowns[kind][i].mean;
    if (g > best) { best = g; bi = i; }
  });
  return { label: labels[bi], best };
}

function setupCanvas(canvas, cssH) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth || 640;
  const h = cssH || canvas.clientHeight || 230;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return { ctx, w, h };
}

function loadImage(src) {
  if (images.has(src)) return images.get(src);
  const pending = new Promise((resolve, reject) => {
    const im = new Image();
    im.onload = () => resolve(im);
    im.onerror = () => reject(new Error(src));
    im.src = src;
  });
  images.set(src, pending);
  return pending;
}

async function main() {
  try {
    const [A, B, C, M] = await Promise.all([
      fetch("assets/a_results.json").then((r) => r.json()),
      fetch("assets/b_results.json").then((r) => r.json()),
      fetch("assets/coverage.json").then((r) => r.json()),
      fetch("assets/demo/manifest.json").then((r) => r.json()),
    ]);
    state.A = A;
    state.B = B;
    state.C = C;
    state.M = M;
    state.confIdx = nearestIdx(A.variants.V2.threshold_curve.conf, A.protocol.conf_operating_point);
    state.budgetIdx = budgetStops().length - 1;
    state.heatScore = B.loco.best_score;
    state.heatClass = Object.keys(B.loco.table)[0];
    bind();
    renderFixed();
    renderA();
    renderB();
    drawSky();
  } catch (e) {
    const p = $("err");
    p.hidden = false;
    p.textContent = "Could not load the result files. " + (e && e.message ? e.message : e);
  }
}

function bind() {
  $("conf").max = state.A.variants.V2.threshold_curve.conf.length - 1;
  $("conf").addEventListener("input", () => {
    state.confIdx = Number($("conf").value);
    renderLive();
    drawConf();
    drawSky();
  });
  $("resetA").addEventListener("click", resetA);
  $("session").addEventListener("change", () => {
    state.session = $("session").value;
    state.frame = 0;
    stopPlay();
    syncFrameSlider();
    drawSky();
  });
  $("frame").addEventListener("input", () => {
    state.frame = Number($("frame").value);
    drawSky();
  });
  $("play").addEventListener("click", togglePlay);
  $("residual").addEventListener("change", () => {
    state.residual = $("residual").checked;
    drawSky();
  });
  $("trails").addEventListener("change", () => {
    state.trails = $("trails").checked;
    drawSky();
  });
  document.querySelectorAll("[data-layer]").forEach((btn) => {
    btn.addEventListener("click", () => setLayer(btn.dataset.layer));
  });
  document.querySelectorAll("#barMode button").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.kind = btn.dataset.kind;
      document.querySelectorAll("#barMode button").forEach((b) => {
        b.setAttribute("aria-pressed", b === btn ? "true" : "false");
      });
      drawBars();
    });
  });
  $("jumpFail").addEventListener("click", () => {
    state.session = "2023_06_20";
    state.frame = 0;
    $("session").value = state.session;
    stopPlay();
    syncFrameSlider();
    drawSky();
    $("sky").scrollIntoView({ behavior: "smooth", block: "center" });
  });
  $("budget").max = budgetStops().length - 1;
  $("budget").addEventListener("input", () => {
    state.budgetIdx = Number($("budget").value);
    renderBudget();
  });
  $("resetB").addEventListener("click", () => {
    state.budgetIdx = budgetStops().length - 1;
    renderBudget();
  });
  const sky = $("sky");
  sky.addEventListener("mousemove", (ev) => {
    const r = sky.getBoundingClientRect();
    pointer.x = (ev.clientX - r.left) / r.width * 256;
    pointer.y = (ev.clientY - r.top) / r.height * 256;
    describePointer();
  });
  sky.addEventListener("mouseleave", () => {
    pointer.x = null;
    $("hover").textContent = "Move over the frame. Rings are ground truth. Filled marks matched within the match radius. Open marks did not.";
  });
  let resizeT;
  window.addEventListener("resize", () => {
    clearTimeout(resizeT);
    resizeT = setTimeout(() => {
      drawConf();
      drawBudget();
      drawHist();
      drawSky();
    }, 80);
  });
  const theme = window.matchMedia("(prefers-color-scheme: light)");
  if (theme.addEventListener) {
    theme.addEventListener("change", () => {
      drawConf();
      drawBars();
      drawSeeds();
      drawBudget();
      drawHist();
      fillHeat();
    });
  }
}

function renderFixed() {
  const A = state.A, B = state.B;
  $("valSessions").textContent = A.protocol.val.join(" and ");
  $("nFrames").textContent = String(A.protocol.n_test_frames);
  $("nRsos").textContent = String(A.protocol.n_test_rsos);
  $("matchPx").textContent = n(A.protocol.match_px, 0);
  $("opConf").textContent = n(A.protocol.conf_operating_point, 2);
  $("starFrac").textContent = n(A.registration.registration_fraction.stars * 100, 1) + "%";
  $("starErr").textContent = n(A.registration.star_fit_err_px.median, 2);
  $("satFrac").textContent = n(A.protocol.peak_dn_saturated_fraction * 100, 0) + "%";
  const overall = A.tracking.per_session.OVERALL;
  $("mota").textContent = n(overall.mota);
  $("idf1").textContent = n(overall.idf1);
  $("switches").textContent = String(overall.num_switches);
  $("nObj").textContent = String(overall.num_unique_objects);
  const mapBits = VARIANTS.map((v) => v + " " + n(A.variants[v].metrics.mAP50_inflated.mean));
  $("mapNote").textContent = mapBits.join(", ");
  $("bNote").textContent = "Headline numbers are the mean ± " + B.protocol.std + ". FPR95 is " + B.protocol.fpr95 + ".";
  const note = B.protocol.note;
  $("dupNote").textContent = note.charAt(0).toUpperCase() + note.slice(1) + ".";
  const sel = $("session");
  Object.keys(state.M.sessions).forEach((s) => sel.appendChild(el("option", { value: s }, s)));
  sel.value = state.session;
  fillAblation();
  drawSeeds();
  fillClosed();
  fillConfusion();
  honestA();
  honestB();
}

function fillAblation() {
  const body = $("ablation");
  body.replaceChildren();
  const metrics = ["P", "R", "F1", "AP"];
  const best = {};
  metrics.forEach((m) => {
    best[m] = Math.max(...VARIANTS.map((v) => state.A.variants[v].metrics[m].mean));
  });
  VARIANTS.forEach((v) => {
    const tr = el("tr");
    tr.appendChild(el("td", { class: "name" }, VLABEL[v]));
    metrics.forEach((m) => {
      const stat = state.A.variants[v].metrics[m];
      const td = el("td", { class: "num" }, pm(stat));
      if (Math.abs(stat.mean - best[m]) < 1e-12) td.classList.add("best");
      tr.appendChild(td);
    });
    body.appendChild(tr);
  });
  const ctx = $("context");
  ctx.replaceChildren();
  state.A.context.forEach((row) => {
    const tr = el("tr", { class: "context" });
    const name = el("td", { class: "name" }, row.method);
    name.appendChild(el("div", { class: "fine" }, row.source));
    tr.appendChild(name);
    tr.appendChild(el("td", { class: "num" }, row.P == null ? "" : n(row.P)));
    tr.appendChild(el("td", { class: "num" }, row.R == null ? "" : n(row.R)));
    tr.appendChild(el("td", { class: "num" }, ""));
    tr.appendChild(el("td", { class: "num" }, row.mAP50 == null ? "" : "mAP50 " + n(row.mAP50)));
    ctx.appendChild(tr);
  });
}

function drawSeeds() {
  const svg = $("seeds");
  svg.replaceChildren();
  const ns = "http://www.w3.org/2000/svg";
  const add = (tag, attrs, text) => {
    const node = document.createElementNS(ns, tag);
    Object.entries(attrs).forEach(([k, v]) => node.setAttribute(k, v));
    if (text != null) node.textContent = text;
    svg.appendChild(node);
    return node;
  };
  const W = 640, left = 176, right = 12;
  const X = (val) => left + Math.max(0, Math.min(1, val)) * (W - left - right);
  VARIANTS.forEach((v, row) => {
    const y = 16 + row * 24;
    add("text", { x: 0, y: y + 4, fill: cssVar("--muted"), "font-size": "13", "font-family": "Source Sans 3, sans-serif" }, VLABEL[v]);
    const stat = state.A.variants[v].metrics.F1;
    stat.values.forEach((val, i) => {
      const c = add("circle", { cx: X(val), cy: y, r: 4.5, fill: VCOLOR[v] });
      const title = document.createElementNS(ns, "title");
      title.textContent = "seed " + i + ": " + n(val);
      c.appendChild(title);
    });
    add("line", {
      x1: X(stat.mean), x2: X(stat.mean), y1: y - 8, y2: y + 8,
      stroke: cssVar("--ink"), "stroke-width": "1.5",
    });
  });
}

function renderA() {
  syncFrameSlider();
  renderLive();
  drawConf();
  drawBars();
}

function renderLive() {
  const idx = state.confIdx;
  const conf = confOf();
  const op = state.A.protocol.conf_operating_point;
  $("conf").value = idx;
  $("confOut").textContent = n(conf, 2);
  $("confState").textContent = Math.abs(conf - op) < 1e-9 ? "Reported point" : "Moved";
  const body = $("live");
  body.replaceChildren();
  const f1s = VARIANTS.map((v) => state.A.variants[v].threshold_curve.F1.mean[idx]);
  const best = Math.max(...f1s);
  VARIANTS.forEach((v) => {
    const tc = state.A.variants[v].threshold_curve;
    const tr = el("tr");
    tr.appendChild(el("td", { class: "name" }, VLABEL[v]));
    ["P", "R", "F1"].forEach((m) => {
      const td = el("td", { class: "num" }, pm({ mean: tc[m].mean[idx], std: tc[m].std[idx] }));
      if (m === "F1" && Math.abs(tc.F1.mean[idx] - best) < 1e-12) td.classList.add("best");
      tr.appendChild(td);
    });
    body.appendChild(tr);
  });
}

function drawConf() {
  const canvas = $("confChart");
  if (!canvas.clientWidth) return;
  const { ctx, w, h } = setupCanvas(canvas, 230);
  const pad = { l: 38, r: 12, t: 14, b: 26 };
  const ink = cssVar("--muted");
  const line = cssVar("--line");
  const X = (c) => pad.l + c * (w - pad.l - pad.r);
  const Y = (v) => pad.t + (1 - Math.max(0, Math.min(1, v))) * (h - pad.t - pad.b);
  ctx.strokeStyle = line;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.l, pad.t);
  ctx.lineTo(pad.l, h - pad.b);
  ctx.lineTo(w - pad.r, h - pad.b);
  ctx.stroke();
  ctx.fillStyle = ink;
  ctx.font = "11px IBM Plex Mono, ui-monospace, monospace";
  [0, 0.5, 1].forEach((g) => {
    ctx.fillText(g.toFixed(1), 2, Y(g) + 4);
    ctx.strokeStyle = line;
    ctx.beginPath();
    ctx.moveTo(pad.l, Y(g));
    ctx.lineTo(w - pad.r, Y(g));
    ctx.stroke();
  });
  VARIANTS.forEach((v) => {
    const tc = state.A.variants[v].threshold_curve;
    ctx.beginPath();
    ctx.strokeStyle = VCOLOR[v];
    ctx.lineWidth = v === "V2" || v === "V3" ? 2.2 : 1.3;
    tc.conf.forEach((c, i) => {
      const x = X(c), y = Y(tc.F1.mean[i]);
      if (i) ctx.lineTo(x, y);
      else ctx.moveTo(x, y);
    });
    ctx.stroke();
  });
  const op = state.A.protocol.conf_operating_point;
  const c = confOf();
  if (Math.abs(c - op) > 1e-9) {
    ctx.setLineDash([2, 3]);
    ctx.strokeStyle = ink;
    ctx.beginPath();
    ctx.moveTo(X(op), pad.t);
    ctx.lineTo(X(op), h - pad.b);
    ctx.stroke();
    ctx.setLineDash([]);
  }
  ctx.strokeStyle = cssVar("--ink");
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(X(c), pad.t);
  ctx.lineTo(X(c), h - pad.b);
  ctx.stroke();
  ctx.fillStyle = ink;
  ctx.fillText("confidence", w - pad.r - 78, h - 6);
  ctx.fillText("F1", 8, 12);
}

function drawBars() {
  const svg = $("bars");
  svg.replaceChildren();
  const ns = "http://www.w3.org/2000/svg";
  const add = (tag, attrs, text) => {
    const node = document.createElementNS(ns, tag);
    Object.entries(attrs).forEach(([k, v]) => node.setAttribute(k, v));
    if (text != null) node.textContent = text;
    svg.appendChild(node);
    return node;
  };
  const kind = state.kind;
  const labels = binLabels(state.A.bins[kind].edges, kind);
  const W = 640, H = 230, left = 36, right = 8, top = 12, bottom = 58;
  const plotW = W - left - right;
  const plotH = H - top - bottom;
  const groupW = plotW / labels.length;
  const barW = Math.min(16, (groupW - 20) / VARIANTS.length);
  const ink = cssVar("--muted");
  const axis = cssVar("--line");
  [0, 0.5, 1].forEach((g) => {
    const y = top + (1 - g) * plotH;
    add("line", { x1: left, x2: W - right, y1: y, y2: y, stroke: axis });
    add("text", { x: 0, y: y + 4, fill: ink, "font-size": "11", "font-family": "IBM Plex Mono, monospace" }, g.toFixed(1));
  });
  labels.forEach((lab, i) => {
    VARIANTS.forEach((v, k) => {
      const stat = state.A.variants[v].breakdowns[kind][i];
      const h = Math.max(0, Math.min(1, stat.mean)) * plotH;
      const x = left + i * groupW + 10 + k * (barW + 2);
      const y = top + plotH - h;
      const rect = add("rect", { x, y, width: barW, height: h, fill: VCOLOR[v] });
      const title = document.createElementNS(ns, "title");
      title.textContent = VLABEL[v] + " " + pm(stat);
      rect.appendChild(title);
      const y0 = top + plotH - Math.max(0, Math.min(1, stat.mean - stat.std)) * plotH;
      const y1 = top + plotH - Math.max(0, Math.min(1, stat.mean + stat.std)) * plotH;
      const cx = x + barW / 2;
      add("line", { x1: cx, x2: cx, y1: y0, y2: y1, stroke: cssVar("--ink"), "stroke-width": "1" });
    });
    add("text", {
      x: left + i * groupW + groupW / 2, y: H - 28, fill: ink, "font-size": "11",
      "text-anchor": "middle", "font-family": "Source Sans 3, sans-serif",
    }, lab);
  });
  const gap = gapSentence(kind);
  const unk = state.A.bins[kind].unknown;
  let note = "Recall, mean over seeds. Whiskers are the seed standard deviation. Largest V2 minus V1 gap in this split: " + gap.label + ", " + n(gap.best) + ".";
  if (unk) note += " " + unk + " objects have no value for this measurement.";
  $("barNote").textContent = note;
}

function honestA() {
  const A = state.A;
  const f = (v) => A.variants[v].metrics.F1;
  const lines = [];
  lines.push("Registration is the gain. V2 F1 is " + pm(f("V2")) + ". V1, the raw pair, is " + pm(f("V1")) + ".");
  const diff = Math.abs(f("V2").mean - f("V3").mean);
  const scatter = Math.max(f("V2").std, f("V3").std);
  lines.push("V3 F1 is " + pm(f("V3")) + ". The gap to V2 is " + n(diff) + ", " + (diff < scatter ? "smaller than the seed scatter, so the star mask is not a gain." : "see the table."));
  lines.push("A single frame does not find these objects. V0 F1 is " + pm(f("V0")) + ".");
  const sess = "2023_06_20";
  const frames = state.M.sessions[sess].frames;
  const nStars = frames.filter((fr) => fr.registration === "stars").length;
  const rec = (v) => pm(A.variants[v].per_session_recall[sess]);
  const v1 = A.variants.V1.per_session_recall[sess].mean;
  const v2 = A.variants.V2.per_session_recall[sess].mean;
  lines.push("On " + sess + ", " + nStars + " of " + frames.length + " frames were aligned from plate-solved stars. Recall there is V1 " + rec("V1") + ", V2 " + rec("V2") + ", V3 " + rec("V3") + "." + (v1 > v2 ? " The raw pair is ahead of the residual on this session." : ""));
  const bright = gapSentence("brightness");
  const speed = gapSentence("speed");
  const pick = bright.best >= speed.best ? { ...bright, name: "brightness" } : { ...speed, name: "speed" };
  lines.push("Among brightness and speed bins, the largest V2 minus V1 recall gap is the " + pick.name + " bin " + pick.label + ", " + n(pick.best) + ".");
  const near = A.variants.V3.breakdowns.star_distance[0].mean - A.variants.V2.breakdowns.star_distance[0].mean;
  if (near < 0) {
    lines.push("In the closest star bin, V3 recall is " + n(Math.abs(near)) + " below V2.");
  }
  const ul = $("honestA");
  ul.replaceChildren();
  lines.forEach((t) => ul.appendChild(el("li", null, t)));
}

function syncFrameSlider() {
  const frames = sessionFrames();
  $("frame").max = frames.length - 1;
  $("frame").value = state.frame;
  $("frameLabel").textContent = (state.frame + 1) + " / " + frames.length;
}

function setLayer(layer) {
  state.layer = layer;
  document.querySelectorAll("[data-layer]").forEach((b) => {
    b.setAttribute("aria-pressed", b.dataset.layer === layer ? "true" : "false");
  });
  drawSky();
}

function stopPlay() {
  state.playing = false;
  $("play").textContent = "Play";
  if (state.timer) clearInterval(state.timer);
  state.timer = null;
}

function togglePlay() {
  if (state.playing) { stopPlay(); return; }
  state.playing = true;
  $("play").textContent = "Pause";
  state.timer = setInterval(() => {
    if (state.frame >= sessionFrames().length - 1) { stopPlay(); return; }
    state.frame += 1;
    $("frame").value = state.frame;
    drawSky();
  }, 140);
}

function matchFrame(gt, dets, thr, px) {
  const order = [];
  dets.forEach((d, i) => { if (d[2] >= thr) order.push(i); });
  order.sort((a, b) => dets[b][2] - dets[a][2]);
  const taken = new Set();
  const detHit = new Set();
  const gtHit = new Set();
  order.forEach((i) => {
    let best = -1, bd = px;
    gt.forEach((g, j) => {
      if (taken.has(j)) return;
      const dist = Math.hypot(g[0] - dets[i][0], g[1] - dets[i][1]);
      if (dist <= bd) { bd = dist; best = j; }
    });
    if (best >= 0) {
      taken.add(best);
      detHit.add(i);
      gtHit.add(best);
    }
  });
  return { detHit, gtHit, shown: order.length };
}

function drawTrails(ctx, frame, S) {
  const pack = state.M.sessions[state.session].tracks;
  if (!pack || !pack.frames) return;
  const idx = pack.frames.indexOf(frame.key);
  if (idx < 0) return;
  ctx.lineWidth = Math.max(1, S * 0.55);
  ctx.strokeStyle = "rgba(224, 154, 60, 0.9)";
  ctx.lineCap = "round";
  Object.values(pack.tracks).forEach((pts) => {
    const tail = [];
    for (let s = Math.max(0, idx - 8); s <= idx; s++) {
      if (pts[String(s)]) tail.push(pts[String(s)]);
    }
    if (tail.length < 2) return;
    ctx.beginPath();
    tail.forEach((p, i) => {
      const x = p[0] * S, y = p[1] * S;
      if (i) ctx.lineTo(x, y);
      else ctx.moveTo(x, y);
    });
    ctx.stroke();
  });
}

async function drawSky() {
  const gen = ++state.skyGen;
  const frames = sessionFrames();
  const frame = frames[state.frame];
  syncFrameSlider();
  const canvas = $("sky");
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth || 560;
  const h = canvas.clientHeight || w;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = "#050505";
  ctx.fillRect(0, 0, w, h);
  const src = "assets/demo/" + (state.residual ? frame.residual : frame.file);
  let im;
  try {
    im = await loadImage(src);
  } catch (e) {
    return;
  }
  if (gen !== state.skyGen) return;
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(im, 0, 0, w, h);
  const S = w / 256;
  const conf = confOf();
  const px = state.A.protocol.match_px;
  const layers = state.layer === "both" ? ["v0", "v3"] : [state.layer];
  if (state.trails) drawTrails(ctx, frame, S);
  const gtHit = new Set();
  const bits = [];
  layers.forEach((layer) => {
    const matched = matchFrame(frame.gt, frame[layer], conf, px);
    matched.gtHit.forEach((j) => gtHit.add(j));
    bits.push(layer.toUpperCase() + " " + matched.detHit.size + " matched, " + (matched.shown - matched.detHit.size) + " not");
    const color = layer === "v0" ? "#9ec7e6" : "#e0681f";
    frame[layer].forEach((d, i) => {
      if (d[2] < conf) return;
      const x = d[0] * S, y = d[1] * S, r = Math.max(3, S * 2.1);
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.strokeStyle = color;
      ctx.lineWidth = Math.max(1, S * 0.45);
      ctx.stroke();
      if (matched.detHit.has(i)) {
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(x, y, r * 0.45, 0, Math.PI * 2);
        ctx.fill();
      }
    });
  });
  frame.gt.forEach((g) => {
    ctx.beginPath();
    ctx.arc(g[0] * S, g[1] * S, Math.max(7, S * 5), 0, Math.PI * 2);
    ctx.strokeStyle = "#ffb000";
    ctx.lineWidth = Math.max(1, S * 0.5);
    ctx.stroke();
  });
  const reg = REG[frame.registration] || frame.registration;
  $("skyMeta").textContent = state.session + "   frame " + frame.frame + "   " + reg + "   " + frame.gt.length + " labels   " + bits.join("   ");
  describePointer();
  for (let i = state.frame + 1; i < Math.min(frames.length, state.frame + 8); i++) {
    loadImage("assets/demo/" + frames[i].file);
  }
}

function describePointer() {
  if (pointer.x == null) return;
  const frame = sessionFrames()[state.frame];
  const conf = confOf();
  const layers = state.layer === "both" ? ["v0", "v3"] : [state.layer];
  let best = null;
  const consider = (d, text) => {
    if (!best || d < best.d) best = { d, text };
  };
  frame.gt.forEach((g) => {
    const d = Math.hypot(g[0] - pointer.x, g[1] - pointer.y);
    if (d <= 10) consider(d, "ground truth at " + n(g[0], 1) + ", " + n(g[1], 1));
  });
  layers.forEach((layer) => {
    frame[layer].forEach((det) => {
      if (det[2] < conf) return;
      const d = Math.hypot(det[0] - pointer.x, det[1] - pointer.y);
      if (d <= 10) consider(d, layer.toUpperCase() + " confidence " + n(det[2], 3) + " at " + n(det[0], 1) + ", " + n(det[1], 1));
    });
  });
  $("hover").textContent = best ? best.text : "x " + n(pointer.x, 1) + "   y " + n(pointer.y, 1);
}

function resetA() {
  state.confIdx = nearestIdx(state.A.variants.V2.threshold_curve.conf, state.A.protocol.conf_operating_point);
  state.session = "2023_03_31";
  state.frame = 0;
  state.residual = false;
  state.trails = false;
  $("residual").checked = false;
  $("trails").checked = false;
  $("session").value = state.session;
  stopPlay();
  setLayer("v3");
  renderLive();
  drawConf();
  drawSky();
}

function fillClosed() {
  const body = $("closed");
  body.replaceChildren();
  const ours = el("tr");
  ours.appendChild(el("td", null, "1D ResNet, this page"));
  ours.appendChild(el("td", { class: "num" }, pm(state.B.closed_set.accuracy)));
  ours.appendChild(el("td", { class: "num" }, pm(state.B.closed_set.macro_f1)));
  ours.appendChild(el("td", null, "3 seeds, this split"));
  body.appendChild(ours);
  state.B.paper.forEach((p) => {
    const tr = el("tr", { class: "context" });
    tr.appendChild(el("td", null, p.method));
    tr.appendChild(el("td", { class: "num" }, n(p.accuracy)));
    tr.appendChild(el("td", { class: "num" }, n(p.macro_f1)));
    tr.appendChild(el("td", null, p.source));
    body.appendChild(tr);
  });
}

function fillConfusion() {
  const table = $("confusion");
  table.replaceChildren();
  const classes = state.B.closed_set.classes;
  const matrix = state.B.closed_set.confusion_seed0;
  const short = (c) => c.replace(" R/B", "");
  const head = el("tr");
  head.appendChild(el("th", null, "True, predicted"));
  classes.forEach((c) => head.appendChild(el("th", { class: "num" }, short(c))));
  table.appendChild(head);
  classes.forEach((c, i) => {
    const tr = el("tr");
    tr.appendChild(el("th", { class: "name" }, short(c)));
    classes.forEach((pred, j) => {
      const td = el("td", {
        class: "num",
        title: c + " predicted as " + pred + ": " + n(matrix[i][j]),
      }, n(matrix[i][j], 2));
      if (i === j) td.classList.add("diag");
      td.style.background = "color-mix(in srgb, var(--v1) " + Math.round(matrix[i][j] * 100) + "%, transparent)";
      tr.appendChild(td);
    });
    table.appendChild(tr);
  });
}

function renderB() {
  const means = state.B.protocol.scores.map((s) => SCORE[s] + " " + n(state.B.loco.mean_auroc_over_classes[s]));
  $("meanScores").textContent = means.join(", ");
  $("bestScore").textContent = SCORE[state.B.loco.best_score];
  fillHeat();
  renderBudget();
  drawHist();
}

function fillHeat() {
  const table = $("heat");
  table.replaceChildren();
  const scores = state.B.protocol.scores;
  const classes = Object.keys(state.B.loco.table);
  const head = el("tr");
  head.appendChild(el("th", null, "Held out"));
  scores.forEach((s) => head.appendChild(el("th", null, SCORE[s])));
  table.appendChild(head);
  classes.forEach((c) => {
    const tr = el("tr");
    tr.appendChild(el("th", { class: "name" }, c.replace(" R/B", "")));
    scores.forEach((s) => {
      const stat = state.B.loco.table[c].auroc[s];
      const btn = el("button", { type: "button" });
      btn.appendChild(document.createTextNode(n(stat.mean, 2)));
      btn.appendChild(el("br"));
      btn.appendChild(document.createTextNode("± " + n(stat.std, 2)));
      const t = Math.max(0, Math.min(1, (stat.mean - 0.3) / 0.7));
      btn.style.background = "color-mix(in srgb, var(--accent) " + Math.round(t * 100) + "%, var(--bg-2))";
      btn.style.color = stat.mean > 0.82 ? "#1c1914" : "var(--ink)";
      if (stat.mean < 0.5) btn.classList.add("under");
      btn.setAttribute("aria-pressed", c === state.heatClass && s === state.heatScore ? "true" : "false");
      btn.addEventListener("click", () => {
        state.heatClass = c;
        state.heatScore = s;
        fillHeat();
        drawHist();
      });
      tr.appendChild(el("td")).appendChild(btn);
      // bug: el("td") appended then appendChild btn on the return? 
      // tr.appendChild returns the td. Then .appendChild(btn) is on the td. Good.
    });
    table.appendChild(tr);
  });
}

function renderBudget() {
  const stops = budgetStops();
  const s = stops[state.budgetIdx];
  const acc = state.B.closed_set.budget[s].accuracy;
  const f1 = state.B.closed_set.budget[s].macro_f1;
  $("budget").value = state.budgetIdx;
  $("budgetState").textContent = s === stops[stops.length - 1] ? "Full series" : "Truncated";
  const bits = state.B.protocol.scores.map((sc) => SCORE[sc] + " " + pm(state.B.loco.budget[s].auroc[sc]));
  $("budgetRead").textContent = s + " s: closed-set accuracy " + pm(acc) + ", macro-F1 " + pm(f1) + ". Unknown AUROC: " + bits.join("; ") + ".";
  $("covNote").textContent = "Median last observed sample is at " + n(state.C.median_last_observed_s, 1) + " s. " + n(state.C.fraction_last_obs_before_600s * 100, 0) + "% of test curves end before 600 s.";
  drawBudget();
}

function drawBudget() {
  const canvas = $("budgetChart");
  if (!canvas.clientWidth) return;
  const { ctx, w, h } = setupCanvas(canvas, 230);
  const stops = budgetStops();
  const gap = 36;
  const pw = (w - gap) / 2;
  drawPanel(ctx, 8, pw, h, stops, "Closed-set accuracy", (s) => state.B.closed_set.budget[s].accuracy, null);
  drawPanel(ctx, 8 + pw + gap, pw, h, stops, "Unknown AUROC", null, state.B.protocol.scores);
}

function drawPanel(ctx, left, width, h, stops, title, one, scores) {
  const top = 22, bottom = 32, right = 8;
  const plotW = width - right;
  const plotH = h - top - bottom;
  const values = [];
  stops.forEach((s) => {
    if (one) values.push(one(s).mean);
    else scores.forEach((sc) => values.push(state.B.loco.budget[s].auroc[sc].mean));
  });
  let lo = Math.min(0.5, ...values) - 0.03;
  let hi = Math.max(...values) + 0.04;
  const X = (i) => left + (stops.length === 1 ? plotW / 2 : (i / (stops.length - 1)) * plotW);
  const Y = (v) => top + (1 - (v - lo) / (hi - lo)) * plotH;
  ctx.fillStyle = cssVar("--muted");
  ctx.font = "12px Source Sans 3, sans-serif";
  ctx.fillText(title, left, 14);
  ctx.strokeStyle = cssVar("--line");
  ctx.beginPath();
  ctx.moveTo(left, top);
  ctx.lineTo(left, top + plotH);
  ctx.lineTo(left + plotW, top + plotH);
  ctx.stroke();
  const series = one ? [{ key: "acc", color: "#5b8db8", stat: one }] : scores.map((sc) => ({
    key: sc,
    color: SCORE_COLOR[sc],
    stat: (s) => state.B.loco.budget[s].auroc[sc],
  }));
  series.forEach((ser) => {
    const thick = !scores || ser.key === state.B.loco.best_score;
    ctx.beginPath();
    ctx.strokeStyle = ser.color;
    ctx.lineWidth = thick ? 2.2 : 1;
    ctx.globalAlpha = thick ? 1 : 0.75;
    stops.forEach((s, i) => {
      const y = Y(ser.stat(s).mean);
      if (i) ctx.lineTo(X(i), y);
      else ctx.moveTo(X(i), y);
    });
    ctx.stroke();
    if (thick) {
      stops.forEach((s, i) => {
        const stat = ser.stat(s);
        ctx.globalAlpha = 1;
        ctx.strokeStyle = ser.color;
        ctx.beginPath();
        ctx.moveTo(X(i), Y(stat.mean - stat.std));
        ctx.lineTo(X(i), Y(stat.mean + stat.std));
        ctx.stroke();
        ctx.fillStyle = ser.color;
        ctx.beginPath();
        ctx.arc(X(i), Y(stat.mean), 3.2, 0, Math.PI * 2);
        ctx.fill();
      });
    }
  });
  ctx.globalAlpha = 1;
  const i = state.budgetIdx;
  ctx.strokeStyle = cssVar("--ink");
  ctx.setLineDash([3, 3]);
  ctx.beginPath();
  ctx.moveTo(X(i), top);
  ctx.lineTo(X(i), top + plotH);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = cssVar("--muted");
  ctx.font = "10px IBM Plex Mono, monospace";
  ctx.textAlign = "center";
  const labelAll = plotW / stops.length > 26;
  stops.forEach((s, k) => {
    if (!labelAll && k !== 0 && k !== stops.length - 1 && k !== state.budgetIdx) return;
    ctx.fillText(s, X(k), h - 8);
  });
  ctx.textAlign = "left";
}

function drawHist() {
  const row = state.B.loco.table[state.heatClass];
  const s = state.heatScore;
  $("heatRead").textContent = state.heatClass + ", " + SCORE[s] + ": AUROC " + pm(row.auroc[s]) + ", FPR95 " + pm(row.fpr95[s]) + ". Known-class accuracy of this leave-one-out model: " + pm(row.known_accuracy) + ".";
  const canvas = $("hist");
  if (!canvas.clientWidth) return;
  const hist = state.B.loco.hist_seed0[state.heatClass][s];
  const { ctx, w, h } = setupCanvas(canvas, 230);
  const bins = hist.known.length;
  const peak = Math.max(...hist.known, ...hist.unknown, 1e-9);
  const left = 8, right = 8, top = 18, bottom = 26;
  const plotW = w - left - right;
  const plotH = h - top - bottom;
  const bw = plotW / bins;
  for (let i = 0; i < bins; i++) {
    const hk = hist.known[i] / peak * plotH;
    const hu = hist.unknown[i] / peak * plotH;
    ctx.fillStyle = "rgba(140,140,140,0.9)";
    ctx.fillRect(left + i * bw, top + plotH - hk, bw * 0.46, hk);
    ctx.fillStyle = "rgba(224,104,31,0.92)";
    ctx.fillRect(left + i * bw + bw * 0.5, top + plotH - hu, bw * 0.46, hu);
  }
  ctx.fillStyle = cssVar("--muted");
  ctx.font = "11px IBM Plex Mono, monospace";
  ctx.fillText(n(hist.edges[0], 1), left, h - 8);
  const end = n(hist.edges[hist.edges.length - 1], 1);
  ctx.fillText(end, Math.max(left, w - right - ctx.measureText(end).width), h - 8);
  ctx.font = "12px Source Sans 3, sans-serif";
  ctx.fillText(SCORE[s] + ", seed 0", left, 13);
}

function honestB() {
  const B = state.B, C = state.C;
  const lines = [];
  lines.push("Closed-set accuracy is " + pm(B.closed_set.accuracy) + ", macro-F1 " + pm(B.closed_set.macro_f1) + ". The paper rows use the same split and their own code, so they are context, not a paired comparison.");
  let best = null, worst = null;
  Object.entries(B.loco.table).forEach(([c, row]) => {
    const a = row.auroc.knn.mean;
    const f = row.fpr95.knn.mean;
    if (!best || a > best.a) best = { c, a };
    if (!worst || f > worst.f) worst = { c, f };
  });
  lines.push("kNN has the highest mean AUROC, " + n(B.loco.mean_auroc_over_classes.knn) + ". Its strongest held-out class is " + best.c + " at " + n(best.a) + ". At 95% known acceptance it still accepts " + n(worst.f, 2) + " of held-out " + worst.c + ".");
  let energyLead = null, mahaLead = null;
  Object.entries(B.loco.table).forEach(([c, row]) => {
    const d = row.auroc.energy.mean - row.auroc.mahalanobis.mean;
    if (!energyLead || d > energyLead.d) energyLead = { c, d };
    if (!mahaLead || -d > mahaLead.d) mahaLead = { c, d: -d };
  });
  lines.push("The scores do not agree. Energy leads Mahalanobis by " + n(energyLead.d) + " on " + energyLead.c + ". Mahalanobis leads energy by " + n(mahaLead.d) + " on " + mahaLead.c + ".");
  lines.push("The median test curve has its last sample at " + n(C.median_last_observed_s, 1) + " s. " + n(C.fraction_last_obs_before_600s * 100, 0) + "% end before 600 s, and " + n(Number(C.fraction_with_any_point_after_s["600"]) * 100, 0) + "% have any sample after 600 s. The flat line from 600 s to 1,000 s is partly the curves running out, not only the classifier having enough.");
  const ul = $("honestB");
  ul.replaceChildren();
  lines.forEach((t) => ul.appendChild(el("li", null, t)));
}

document.addEventListener("DOMContentLoaded", main);
