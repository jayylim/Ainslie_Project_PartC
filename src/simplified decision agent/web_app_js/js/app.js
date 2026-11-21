import { apiFetch } from "./session_api.js";

const DEFAULT_CONFIG = {
  ss_value: 3,
  ss_delay: 0,
  ll_value: 20,
  ll_delay: 12,
  length: null,
  horizon: 10,
  discount_factor: 0.6,
  max_steps: null,
  vision: "long",
};
const DEFAULT_VISION = DEFAULT_CONFIG.vision;
const DEFAULT_DELAY_MAX = 24;
const DEFAULT_DELAY_STEP = 4;
const DEFAULT_STICKINESS = 0.6;
const DEFAULT_LEARN_STICKINESS = true;
const DEFAULT_PRIOR_ALPHA = 1;
const DEFAULT_PRIOR_BETA = 1;
const DEFAULT_PRIOR_MEAN = 0.5;
const DEFAULT_STICKY_STEP_SEED = "";

let vision = DEFAULT_VISION;
let learnStickiness = !!DEFAULT_LEARN_STICKINESS;
let section = "steppable";
let stickyClickBound = false;
let configTimer = null;
let stickyOptsTimer = null;
let finished = false;

const metaEl = document.getElementById("meta");
const metaGraphEl = document.getElementById("metaGraph");
const metaSailsEl = document.getElementById("metaSails");
const singleViewEl = document.getElementById("singleView");
const stickyStackEl = document.getElementById("stickyStack");
const stickyStepOptsEl = document.getElementById("stickyStepOpts");
const ssValueEl = document.getElementById("ssValue");
const ssDelayEl = document.getElementById("ssDelay");
const llValueEl = document.getElementById("llValue");
const llDelayEl = document.getElementById("llDelay");
const lengthEl = document.getElementById("length");
const horizonEl = document.getElementById("horizon");
const discountFactorEl = document.getElementById("discountFactor");
const maxStepsEl = document.getElementById("maxSteps");
const visionShort = document.getElementById("visionShort");
const visionLong = document.getElementById("visionLong");
const learnStickinessOff = document.getElementById("learnStickinessOff");
const learnStickinessOn = document.getElementById("learnStickinessOn");
const stickyStepPriorAlphaEl = document.getElementById("stickyStepPriorAlpha");
const stickyStepPriorBetaEl = document.getElementById("stickyStepPriorBeta");
const stickyStepSeedEl = document.getElementById("stickyStepSeed");
const stickyStepPriorMeanEl = document.getElementById("stickyStepPriorMean");
const demoSelect = document.getElementById("demoSelect");
const steppableDemoSelect = document.getElementById("steppableDemoSelect");
const delayMaxEl = document.getElementById("delayMax");
const delayStepEl = document.getElementById("delayStep");
const stickinessEl = document.getElementById("stickiness");
const btnStep = document.getElementById("btnStep");
const btnStepBack = document.getElementById("btnStepBack");
const btnReset = document.getElementById("btnReset");
const btnResetDefaults = document.getElementById("btnResetDefaults");
const stepHint = document.getElementById("stepHint");

function optionalToInput(value) {
  return value == null || value === "" ? "" : String(value);
}

function applyDefaultsToForm() {
  ssValueEl.value = DEFAULT_CONFIG.ss_value;
  ssDelayEl.value = DEFAULT_CONFIG.ss_delay;
  llValueEl.value = DEFAULT_CONFIG.ll_value;
  llDelayEl.value = DEFAULT_CONFIG.ll_delay;
  lengthEl.value = optionalToInput(DEFAULT_CONFIG.length);
  horizonEl.value = optionalToInput(DEFAULT_CONFIG.horizon);
  discountFactorEl.value = DEFAULT_CONFIG.discount_factor;
  maxStepsEl.value = optionalToInput(DEFAULT_CONFIG.max_steps);
  vision = DEFAULT_CONFIG.vision;
  learnStickiness = !!DEFAULT_LEARN_STICKINESS;
  stickyStepPriorAlphaEl.value = String(DEFAULT_PRIOR_ALPHA);
  stickyStepPriorBetaEl.value = String(DEFAULT_PRIOR_BETA);
  stickyStepSeedEl.value = optionalToInput(DEFAULT_STICKY_STEP_SEED);
  syncStickyPriorMeanLabel(DEFAULT_PRIOR_MEAN);
  syncVisionButtons();
  syncLearnStickinessButtons();
  clearConfigError();
}

function steppableDemo() {
  return steppableDemoSelect.value || "visualizing_main";
}

function syncStepHint() {
  const demo = steppableDemo();
  stickyStepOptsEl.classList.toggle("hidden", demo !== "sticky_step");
  btnStepBack.classList.toggle("hidden", demo !== "hyperbolic_ss_ll");
  if (demo === "visualizing_main") {
    stepHint.textContent =
      "Step advances one decision day in the environment";
    } else if (demo === "sticky_step") {
    stepHint.textContent =
      "Step advances one decision day in the environment, decision outcome (stick/switch) is based on perceived stickiness";
  } else {
    stepHint.textContent =
      "Step moves the reward one hour closer";
  }
}
syncStepHint();

function syncVisionButtons() {
  visionShort.classList.toggle("active", vision === "short");
  visionLong.classList.toggle("active", vision === "long");
}
syncVisionButtons();

function syncLearnStickinessButtons() {
  learnStickinessOff.classList.toggle("active", !learnStickiness);
  learnStickinessOn.classList.toggle("active", learnStickiness);
}
syncLearnStickinessButtons();

function parseOptionalInt(raw) {
  const s = String(raw).trim();
  if (s === "") return null;
  const n = parseInt(s, 10);
  return Number.isFinite(n) ? n : null;
}

function currentConfig() {
  return {
    ss_value: parseFloat(ssValueEl.value),
    ss_delay: parseInt(ssDelayEl.value, 10),
    ll_value: parseFloat(llValueEl.value),
    ll_delay: parseInt(llDelayEl.value, 10),
    length: parseOptionalInt(lengthEl.value),
    horizon: parseOptionalInt(horizonEl.value),
    discount_factor: parseFloat(discountFactorEl.value),
    max_steps: parseOptionalInt(maxStepsEl.value),
    vision: vision,
  };
}

function clearConfigError() {
  const errEl = document.getElementById("configError");
  errEl.textContent = "";
  errEl.classList.remove("visible");
  [lengthEl, horizonEl, discountFactorEl, maxStepsEl].forEach((el) =>
    el.classList.remove("field-error")
  );
}

function showConfigError(message, fields) {
  const errEl = document.getElementById("configError");
  errEl.textContent = message;
  errEl.classList.add("visible");
  const f = fields || [];
  lengthEl.classList.toggle("field-error", f.includes("length"));
  horizonEl.classList.toggle("field-error", f.includes("horizon"));
  discountFactorEl.classList.toggle("field-error", f.includes("discount_factor"));
  maxStepsEl.classList.toggle("field-error", f.includes("max_steps"));
}

function syncViewPanes() {
  const sticky = section === "sticky";
  singleViewEl.classList.toggle("hidden", sticky);
  stickyStackEl.classList.toggle("hidden", !sticky);
}

function setSection(name) {
  section = name;
  document.querySelectorAll(".tabs button").forEach((b) => {
    b.classList.toggle("active", b.dataset.section === name);
  });
  document.getElementById("chromeSteppable").classList.toggle("hidden", name !== "steppable");
  document.getElementById("chromeNon").classList.toggle("hidden", name !== "nonsteppable");
  document.getElementById("chromeSticky").classList.toggle("hidden", name !== "sticky");
  syncViewPanes();
  refresh();
}

document.querySelectorAll(".tabs button").forEach((b) => {
  b.addEventListener("click", () => setSection(b.dataset.section));
});

visionShort.addEventListener("click", () => {
  vision = "short";
  syncVisionButtons();
  scheduleConfig();
});
visionLong.addEventListener("click", () => {
  vision = "long";
  syncVisionButtons();
  scheduleConfig();
});

learnStickinessOff.addEventListener("click", () => {
  learnStickiness = false;
  syncLearnStickinessButtons();
  scheduleStickyOptions();
});
learnStickinessOn.addEventListener("click", () => {
  learnStickiness = true;
  syncLearnStickinessButtons();
  scheduleStickyOptions();
});

function parseStickinessDraft(raw) {
  // Allow in-progress decimals ("0.", ".", "") without rewriting the input.
  const s = String(raw).trim();
  if (s === "" || s === "." || s === "-." || s.endsWith(".")) return null;
  const n = Number(s);
  if (!Number.isFinite(n)) return null;
  return Math.max(0, Math.min(1, Math.round(n * 100) / 100));
}

function parsePositiveDraft(raw) {
  const s = String(raw).trim();
  if (s === "" || s === "." || s.endsWith(".")) return null;
  const n = Number(s);
  if (!Number.isFinite(n) || n <= 0) return null;
  return n;
}

function clampPositive(raw, fallback) {
  const n = Number(raw);
  if (!Number.isFinite(n) || n <= 0) return fallback;
  return n;
}

function syncStickyPriorMeanLabel(mean) {
  const m = Number(mean);
  const text = Number.isFinite(m) ? m.toFixed(3) : "—";
  stickyStepPriorMeanEl.textContent = `fixed s=α/(α+β)=${text}`;
}

function parseOptionalIntDraft(raw) {
  const s = String(raw).trim();
  if (s === "") return null;
  if (s === "-" || s.endsWith(".")) return undefined; // still typing
  const n = Number(s);
  if (!Number.isFinite(n) || !Number.isInteger(n)) return undefined;
  return n;
}

function normalizeOptionalInt(raw) {
  const s = String(raw).trim();
  if (s === "") return null;
  const n = Number(s);
  if (!Number.isFinite(n) || !Number.isInteger(n)) return null;
  return n;
}

async function pushStickyOptions({ normalizeInputs = false } = {}) {
  let prior_alpha = parsePositiveDraft(stickyStepPriorAlphaEl.value);
  let prior_beta = parsePositiveDraft(stickyStepPriorBetaEl.value);
  let seed = parseOptionalIntDraft(stickyStepSeedEl.value);
  if (normalizeInputs) {
    prior_alpha = clampPositive(stickyStepPriorAlphaEl.value, DEFAULT_PRIOR_ALPHA);
    prior_beta = clampPositive(stickyStepPriorBetaEl.value, DEFAULT_PRIOR_BETA);
    seed = normalizeOptionalInt(stickyStepSeedEl.value);
    stickyStepPriorAlphaEl.value = String(prior_alpha);
    stickyStepPriorBetaEl.value = String(prior_beta);
    stickyStepSeedEl.value = optionalToInput(seed);
  } else if (prior_alpha == null || prior_beta == null || seed === undefined) {
    // Still typing a decimal / incomplete seed — wait for a complete value.
    return null;
  }
  syncStickyPriorMeanLabel(prior_alpha / (prior_alpha + prior_beta));
  const res = await apiFetch("/api/steppable/sticky_options", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      learn_stickiness: learnStickiness,
      prior_alpha,
      prior_beta,
      seed,
    }),
  });
  if (!res.ok) throw new Error("HTTP " + res.status);
  const payload = await res.json();
  if (payload.options && payload.options.perceived_s != null) {
    syncStickyPriorMeanLabel(payload.options.perceived_s);
  }
  draw(payload);
  return payload;
}

function scheduleStickyOptions() {
  if (steppableDemo() !== "sticky_step" || section !== "steppable") return;
  clearTimeout(stickyOptsTimer);
  stickyOptsTimer = setTimeout(async () => {
    try {
      await pushStickyOptions();
    } catch (err) {
      metaEl.textContent = "Sticky options failed: " + err;
    }
  }, 40);
}

stickyStepPriorAlphaEl.addEventListener("input", scheduleStickyOptions);
stickyStepPriorBetaEl.addEventListener("input", scheduleStickyOptions);
stickyStepSeedEl.addEventListener("input", scheduleStickyOptions);
stickyStepPriorAlphaEl.addEventListener("change", () => {
  pushStickyOptions({ normalizeInputs: true }).catch((err) => {
    metaEl.textContent = "Sticky options failed: " + err;
  });
});
stickyStepPriorBetaEl.addEventListener("change", () => {
  pushStickyOptions({ normalizeInputs: true }).catch((err) => {
    metaEl.textContent = "Sticky options failed: " + err;
  });
});
stickyStepSeedEl.addEventListener("change", () => {
  pushStickyOptions({ normalizeInputs: true }).catch((err) => {
    metaEl.textContent = "Sticky options failed: " + err;
  });
});

async function pushConfig() {
  const cfg = currentConfig();
  // Client-side guard so both-none fails immediately with highlighted fields
  if (cfg.length === null && cfg.horizon === null) {
    const msg =
      "Env length and agent horizon cannot both be none — set at least one.";
    showConfigError(msg, ["length", "horizon"]);
    const err = new Error(msg);
    err.configError = true;
    throw err;
  }
  const res = await apiFetch("/api/config", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(cfg),
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    const msg = body.error || ("config HTTP " + res.status);
    showConfigError(msg, body.fields || ["length", "horizon"]);
    const err = new Error(msg);
    err.configError = true;
    throw err;
  }
  clearConfigError();
  finished = false;
  btnStep.disabled = false;
}

function clampStickiness(raw) {
  let s = Number(raw);
  if (!Number.isFinite(s)) s = DEFAULT_STICKINESS;
  s = Math.max(0, Math.min(1, s));
  return Math.round(s * 100) / 100;
}

function setStickiness(raw) {
  stickinessEl.value = String(clampStickiness(raw));
}

function plotOpts() {
  return {
    responsive: true,
    displayModeBar: true,
    displaylogo: false,
    scrollZoom: false,
    modeBarButtonsToRemove: ["lasso2d", "select2d"],
  };
}

function baseLayout(payload, uirevision) {
  const layout = Object.assign(
    {
      font: {
        family: "Iowan Old Style, Palatino Linotype, Georgia, serif",
        size: Math.round(13 * UI_SCALE),
      },
      paper_bgcolor: "#fff",
      plot_bgcolor: "#fafbfc",
    },
    payload.layout || {}
  );
  if (!layout.uirevision) {
    layout.uirevision = uirevision;
  }
  // Match ~80% UI scale without CSS zoom (keeps Plotly clicks working).
  if (typeof layout.height === "number") {
    layout.height = Math.round(layout.height * UI_SCALE);
  }
  return layout;
}

function fillMeta(el, payload) {
  const m = payload.meta || {};
  if (payload.kind === "sticky") {
    el.textContent =
      `delays = [${(m.delays || []).join(", ")}]h · LL = ${m.reward_LL} · ` +
      `horizon = ${m.horizon} · discount_factor = ${m.discount_factor}`;
  } else if (payload.kind === "sticky_step") {
    const av = m.action_values || {};
    const dayBit = m.day != null ? `Day ${m.day} · ` : "";
    const learnBit = m.learn_stickiness ? "learn On" : "learn Off";
    const sHat =
      m.s_hat != null ? Number(m.s_hat).toFixed(3) : "—";
    const perc =
      m.perceived_s != null ? Number(m.perceived_s).toFixed(3) : "—";
    const priorMean =
      m.prior_mean != null ? Number(m.prior_mean).toFixed(3) : "—";
    let rollBit = "";
    if (m.last_realized != null) {
      const stuckLabel =
        m.last_stuck == null ? "free" : m.last_stuck ? "stick" : "switch";
      const aligned =
        m.last_aligned != null
          ? !!m.last_aligned
          : m.last_realized === m.last_intent;
      const alignBit = aligned
        ? '<span style="color:#2e7d32;font-weight:600;">aligned</span>'
        : '<span style="color:#c62828;font-weight:600;">diverged</span>';
      rollBit =
        ` · last intent=${m.last_intent} → realized=${m.last_realized} (${stuckLabel}, ${alignBit})`;
    }
    const doneBit = m.finished ? " · finished" : "";
    el.innerHTML =
      dayBit +
      `${learnBit} · foresight s=${perc} · ŝ=${sHat} · prior mean=${priorMean} · ` +
      `α=${Number(m.alpha).toFixed(1)} β=${Number(m.beta).toFixed(1)} · ` +
      `<span class="choice">would choose ${m.choice}</span>` +
      ` · dessert = ${Number(av.dessert).toFixed(4)} · resist = ${Number(av.resist).toFixed(4)}` +
      rollBit +
      doneBit;
  } else if (m.action_values) {
    const av = m.action_values;
    const dayBit = m.day != null ? `Day ${m.day} · ` : "";
    const stickyBit =
      m.perceived_s != null
        ? `Perceived Stickiness (s) = ${Number(m.perceived_s).toFixed(2)} · `
        : "";
    const doneBit = m.finished ? " · finished" : "";
    el.innerHTML =
      dayBit +
      stickyBit +
      `<span class="choice">Agent chooses ${m.choice}</span>` +
      ` · dessert = ${av.dessert.toFixed(4)} · resist = ${av.resist.toFixed(4)}` +
      doneBit;
  } else if (payload.kind === "hyperbolic_curve") {
    const doneBit = m.finished ? " · finished" : "";
    el.innerHTML =
      `elapsed = ${m.elapsed}h · remaining = ${m.remaining}h · ` +
      `<span class="choice">V(day 0) = ${m.perceived.toFixed(4)}</span>` +
      ` · k = ${m.discount_factor}` +
      doneBit;
  } else if (payload.kind === "hyperbolic_ss_ll") {
    const doneBit = m.finished ? " · finished" : "";
    const pref =
      m.preference === "indifferent"
        ? "indifferent"
        : m.preference === "none"
          ? "both delivered"
          : "prefers " + m.preference;
    const vss =
      m.v_ss == null ? "delivered" : Number(m.v_ss).toFixed(4);
    const vll =
      m.v_ll == null ? "delivered" : Number(m.v_ll).toFixed(4);
    el.innerHTML =
      `elapsed = ${m.elapsed}h · SS rem = ${m.remaining_ss}h · LL rem = ${m.remaining_ll}h · ` +
      `<span class="choice">${pref}</span>` +
      ` · V_SS = ${vss} · V_LL = ${vll}` +
      ` k=${m.discount_factor}` +
      doneBit;
  } else if (m.perceived != null) {
    el.textContent = `perceived at t=0 = ${m.perceived.toFixed(4)}`;
  } else {
    el.textContent = "";
  }
  if (m.finished != null) {
    finished = !!m.finished;
    btnStep.disabled = finished;
  }
  if (payload.kind === "hyperbolic_ss_ll") {
    btnStepBack.disabled = !(m.elapsed > 0);
  }
}

const UI_SCALE =
  Number.parseFloat(
    getComputedStyle(document.documentElement).getPropertyValue("--ui-scale")
  ) || 0.8;

function onStickyGraphClick(ev) {
  if (section !== "sticky") return;
  const pt = ev.points && ev.points[0];
  if (!pt || pt.x == null || !Number.isFinite(Number(pt.x))) return;
  // Any curve: x is stickiness s on the shared axis (same as clicking the black curve).
  setStickiness(pt.x);
  refreshStickySailsOnly();
}

function ensureStickyClickHandler() {
  const chart = document.getElementById("chartGraph");
  if (!chart) return;
  // Re-bind after every plot: newPlot can drop listeners.
  if (stickyClickBound) {
    if (typeof chart.removeListener === "function") {
      try {
        chart.removeListener("plotly_click", onStickyGraphClick);
      } catch (_err) {
        /* ignore */
      }
    }
    chart.removeEventListener("plotly_click", onStickyGraphClick);
  }
  if (typeof chart.on === "function") {
    chart.on("plotly_click", onStickyGraphClick);
  }
  chart.addEventListener("plotly_click", onStickyGraphClick);
  stickyClickBound = true;
}

const plotReady = { chart: false, chartGraph: false, chartSails: false };

function drawInto(divId, payload) {
  if (typeof Plotly === "undefined") {
    throw new Error("Plotly failed to load (CDN blocked?). Refresh or check network.");
  }
  const uirevision =
    divId +
    "|" +
    JSON.stringify(currentConfig()) +
    "|" +
    (section === "steppable"
      ? steppableDemo()
      : section === "sticky"
        ? divId === "chartSails"
          ? "sticky-sails|" + stickinessEl.value
          : "sticky-graph|" + delayMaxEl.value + "|" + delayStepEl.value
        : demoSelect.value || "");
  const layout = baseLayout(payload, uirevision);
  const opts = plotOpts();
  const done = !plotReady[divId]
    ? Plotly.newPlot(divId, payload.traces, layout, opts)
    : Plotly.react(divId, payload.traces, layout, opts);
  plotReady[divId] = true;
  return done;
}

function draw(payload) {
  try {
    drawInto("chart", payload);
    fillMeta(metaEl, payload);
  } catch (err) {
    metaEl.textContent = String(err.message || err);
  }
}

function drawStickyPair(graphPayload, sailsPayload) {
  try {
    Promise.all([
      drawInto("chartGraph", graphPayload),
      drawInto("chartSails", sailsPayload),
    ])
      .then(() => {
        ensureStickyClickHandler();
        fillMeta(metaGraphEl, graphPayload);
        fillMeta(metaSailsEl, sailsPayload);
      })
      .catch((err) => {
        metaGraphEl.textContent = String(err.message || err);
      });
  } catch (err) {
    metaGraphEl.textContent = String(err.message || err);
  }
}

async function fetchJson(url) {
  const res = await apiFetch(url);
  if (!res.ok) throw new Error("HTTP " + res.status);
  return res.json();
}

async function refreshStickySailsOnly() {
  try {
    const s = parseStickinessDraft(stickinessEl.value);
    if (s == null) return;
    const sails = await fetchJson(
      "/api/demo/visualise_sticky_sails?stickiness=" + encodeURIComponent(s)
    );
    drawInto("chartSails", sails);
    fillMeta(metaSailsEl, sails);
  } catch (err) {
    metaSailsEl.textContent = "Failed to load sails: " + err;
  }
}

async function refresh() {
  try {
    if (section === "sticky") {
      const draft = parseStickinessDraft(stickinessEl.value);
      const s = draft != null ? draft : clampStickiness(stickinessEl.value);
      const [graph, sails] = await Promise.all([
        fetchJson(
          `/api/sticky?delay_max=${delayMaxEl.value}` +
            `&delay_step=${delayStepEl.value}`
        ),
        fetchJson(
          "/api/demo/visualise_sticky_sails?stickiness=" + encodeURIComponent(s)
        ),
      ]);
      drawStickyPair(graph, sails);
      return;
    }
    let url;
    if (section === "steppable") {
      url = "/api/steppable/frame?demo=" + encodeURIComponent(steppableDemo());
    } else {
      url = "/api/demo/" + demoSelect.value;
    }
    draw(await fetchJson(url));
  } catch (err) {
    const el = section === "sticky" ? metaGraphEl : metaEl;
    el.textContent = "Failed to load: " + err;
  }
}

async function applyConfigAndRefresh() {
  try {
    await pushConfig();
    await refresh();
  } catch (err) {
    if (!err.configError) {
      metaEl.textContent = "Config failed: " + err;
    } else {
      metaEl.textContent = "";
    }
  }
}

function scheduleConfig() {
  clearTimeout(configTimer);
  configTimer = setTimeout(applyConfigAndRefresh, 40);
}

[ssValueEl, ssDelayEl, llValueEl, llDelayEl, lengthEl, horizonEl, discountFactorEl, maxStepsEl].forEach((el) => {
  el.addEventListener("input", scheduleConfig);
});

demoSelect.addEventListener("change", () => {
  if (section === "nonsteppable") refresh();
});

steppableDemoSelect.addEventListener("change", async () => {
  syncStepHint();
  finished = false;
  btnStep.disabled = false;
  try {
    if (steppableDemo() === "sticky_step") {
      // Apply live learn / prior knobs, then reset the sticky run
      await apiFetch("/api/steppable/sticky_options", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          learn_stickiness: learnStickiness,
          prior_alpha: clampPositive(stickyStepPriorAlphaEl.value, DEFAULT_PRIOR_ALPHA),
          prior_beta: clampPositive(stickyStepPriorBetaEl.value, DEFAULT_PRIOR_BETA),
          seed: normalizeOptionalInt(stickyStepSeedEl.value),
        }),
      });
    }
    const res = await apiFetch("/api/steppable/reset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ demo: steppableDemo() }),
    });
    if (!res.ok) throw new Error("HTTP " + res.status);
    draw(await res.json());
  } catch (err) {
    metaEl.textContent = "Failed to switch demo: " + err;
  }
});

function syncStickyLabels() {
  document.getElementById("delayMaxVal").textContent = delayMaxEl.value + "h";
  document.getElementById("delayStepVal").textContent = delayStepEl.value + "h";
}
delayMaxEl.addEventListener("input", () => {
  syncStickyLabels();
  if (section === "sticky") {
    clearTimeout(configTimer);
    configTimer = setTimeout(refresh, 40);
  }
});
delayStepEl.addEventListener("input", () => {
  syncStickyLabels();
  if (section === "sticky") {
    clearTimeout(configTimer);
    configTimer = setTimeout(refresh, 40);
  }
});
stickinessEl.addEventListener("input", () => {
  if (section === "sticky") {
    clearTimeout(configTimer);
    configTimer = setTimeout(refreshStickySailsOnly, 40);
  }
});
stickinessEl.addEventListener("change", () => {
  // Normalize only on commit (blur/Enter), not while typing decimals.
  setStickiness(stickinessEl.value);
  if (section === "sticky") refreshStickySailsOnly();
});
syncStickyLabels();
syncViewPanes();

btnReset.addEventListener("click", async () => {
  try {
    await pushConfig();
    const res = await apiFetch("/api/steppable/reset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ demo: steppableDemo() }),
    });
    if (!res.ok) throw new Error("HTTP " + res.status);
    draw(await res.json());
    finished = false;
    btnStep.disabled = false;
  } catch (err) {
    if (!err.configError) metaEl.textContent = "Reset failed: " + err;
  }
});

btnResetDefaults.addEventListener("click", async () => {
  clearTimeout(configTimer);
  clearTimeout(stickyOptsTimer);
  applyDefaultsToForm();
  setStickiness(DEFAULT_STICKINESS);
  try {
    await pushConfig();
    if (steppableDemo() === "sticky_step") {
      await pushStickyOptions();
    } else {
      await refresh();
    }
  } catch (err) {
    if (!err.configError) metaEl.textContent = "Reset defaults failed: " + err;
  }
});

btnStep.addEventListener("click", async () => {
  try {
    const res = await apiFetch("/api/steppable/step", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ demo: steppableDemo() }),
    });
    if (!res.ok) throw new Error("HTTP " + res.status);
    draw(await res.json());
  } catch (err) {
    metaEl.textContent = "Step failed: " + err;
  }
});

btnStepBack.addEventListener("click", async () => {
  try {
    const res = await apiFetch("/api/steppable/step_back", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ demo: steppableDemo() }),
    });
    if (!res.ok) throw new Error("HTTP " + res.status);
    draw(await res.json());
  } catch (err) {
    metaEl.textContent = "Step back failed: " + err;
  }
});

refresh();
