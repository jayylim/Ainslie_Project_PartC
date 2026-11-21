/**
 * Client-side port of demo_web_app.py Flask routes + session state.
 * Each browser tab owns its own state (no shared server session).
 */

import {
  DEFAULT_CONFIG,
  DEFAULT_STICKINESS,
  DEFAULT_LEARN_STICKINESS,
  DEFAULT_PRIOR_ALPHA,
  DEFAULT_PRIOR_BETA,
  DEFAULT_PRIOR_MEAN,
  DEFAULT_STICKY_STEP_SEED,
  DEFAULT_DELAY_MAX,
  DEFAULT_DELAY_STEP,
  envParams,
} from "./configs.js";
import { HypAgent } from "./hyperbolic.js";
import { StickyHypAgent } from "./sticky_agent.js";
import { TimelineEnvironment } from "./timeline.js";
import { StickyTimelineEnvironment } from "./sticky_timeline.js";
import * as plot_series from "./plot_series.js";

const _decision_interval = envParams.decision_interval;
const _start = envParams.start;

let _config = structuredClone(DEFAULT_CONFIG);

const _session = {
  environment: null,
  agent: null,
  state: null,
  day: 0,
  finished: false,
};

const _hyp_session = { elapsed: 0, finished: false };
const _hyp_ss_ll_session = { elapsed: 0, finished: false };

const _sticky_step_opts = {
  learn_stickiness: DEFAULT_LEARN_STICKINESS,
  prior_alpha: DEFAULT_PRIOR_ALPHA,
  prior_beta: DEFAULT_PRIOR_BETA,
  perceived_s: DEFAULT_PRIOR_MEAN,
  seed: DEFAULT_STICKY_STEP_SEED,
};

const _sticky_session = {
  environment: null,
  agent: null,
  state: null,
  day: 0,
  finished: false,
  last_info: null,
};

function parseOptionalInt(value) {
  if (value == null || value === "") return null;
  return Math.trunc(Number(value));
}

function buildEnvParams(cfg) {
  return {
    length: parseOptionalInt(cfg.length),
    start: _start,
    decision_interval: _decision_interval,
    rewards: {
      SS: { value: Number(cfg.ss_value), delay: Math.trunc(cfg.ss_delay) },
      LL: { value: Number(cfg.ll_value), delay: Math.trunc(cfg.ll_delay) },
    },
  };
}

function buildAgentParams(cfg) {
  const horizon = parseOptionalInt(cfg.horizon);
  if (horizon === 0) {
    throw new Error("Agent horizon cannot be 0 (use blank for none)");
  }
  return {
    discount_factor: Number(cfg.discount_factor),
    vision: cfg.vision,
    horizon,
  };
}

function buildStickyAgentParams(cfg = null) {
  const base = buildAgentParams(cfg || _config);
  return {
    ...base,
    perceived_s: Number(_sticky_step_opts.perceived_s),
    learn_stickiness: Boolean(_sticky_step_opts.learn_stickiness),
  };
}

export function validateConfig(cfg) {
  const length = parseOptionalInt(cfg.length);
  const horizon = parseOptionalInt(cfg.horizon);
  const max_steps_val = parseOptionalInt(cfg.max_steps);
  if (length == null && horizon == null) {
    return (
      "Env length and agent horizon cannot both be none — " +
      "set at least one (infinite timeline needs a finite horizon)."
    );
  }
  if (horizon === 0) {
    return "Agent horizon cannot be 0 (leave blank for none, or use ≥ 1).";
  }
  if (max_steps_val === 0) {
    return "Max steps cannot be 0 (leave blank for none, or use ≥ 1).";
  }
  const k = Number(cfg.discount_factor);
  if (!Number.isFinite(k)) return "Discount factor must be a number.";
  if (k < 0) return "Discount factor must be ≥ 0.";
  return null;
}

function resetHypSession() {
  Object.assign(_hyp_session, { elapsed: 0, finished: false });
  return _hyp_session;
}

function resetHypSsLlSession() {
  Object.assign(_hyp_ss_ll_session, { elapsed: 0, finished: false });
  return _hyp_ss_ll_session;
}

function syncStickyPresetFromPrior() {
  const a = Number(_sticky_step_opts.prior_alpha);
  const b = Number(_sticky_step_opts.prior_beta);
  _sticky_step_opts.perceived_s = a / (a + b);
}

function rebuildStickySession() {
  syncStickyPresetFromPrior();
  const env_kwargs = {
    ...buildEnvParams(_config),
    prior_alpha: Number(_sticky_step_opts.prior_alpha),
    prior_beta: Number(_sticky_step_opts.prior_beta),
    seed: _sticky_step_opts.seed,
  };
  const environment = new StickyTimelineEnvironment(env_kwargs);
  const agent = new StickyHypAgent(buildStickyAgentParams(_config));
  const state = environment.reset();
  Object.assign(_sticky_session, {
    environment,
    agent,
    state,
    day: 0,
    finished: false,
    last_info: null,
  });
  return _sticky_session;
}

export function rebuildSession(cfg = null) {
  if (cfg != null) Object.assign(_config, cfg);
  const environment = new TimelineEnvironment(buildEnvParams(_config));
  const agent = new HypAgent(buildAgentParams(_config));
  const state = environment.reset();
  Object.assign(_session, {
    environment,
    agent,
    state,
    day: 0,
    finished: false,
  });
  resetHypSession();
  resetHypSsLlSession();
  rebuildStickySession();
  return _session;
}

function freshSnapshot() {
  const environment = new TimelineEnvironment(buildEnvParams(_config));
  const agent = new HypAgent(buildAgentParams(_config));
  const state = environment.reset();
  return [environment, agent, state];
}

function currentSailsFrame() {
  const frame = plot_series.sails_frame(
    _session.agent,
    _session.environment,
    _session.state,
    { day: _session.day }
  );
  frame.meta.finished = _session.finished;
  return frame;
}

function currentHypFrame() {
  const frame = plot_series.hyperbolic_frame(
    _config.ll_value,
    _config.ll_delay,
    _config.discount_factor,
    { elapsed: _hyp_session.elapsed }
  );
  frame.meta.finished = _hyp_session.finished;
  return frame;
}

function hypSsLlMaxDelay() {
  return Math.max(Math.trunc(_config.ss_delay), Math.trunc(_config.ll_delay));
}

function currentHypSsLlFrame() {
  const frame = plot_series.hyperbolic_ss_ll_frame(
    _config.ss_value,
    _config.ll_value,
    _config.ll_delay,
    _config.discount_factor,
    { elapsed: _hyp_ss_ll_session.elapsed, ss_delay: _config.ss_delay }
  );
  frame.meta.finished = _hyp_ss_ll_session.finished;
  return frame;
}

function currentStickyStepFrame() {
  if (_sticky_session.environment == null) rebuildStickySession();
  const agent = _sticky_session.agent;
  agent.learn_stickiness = Boolean(_sticky_step_opts.learn_stickiness);
  agent.perceived_s = Number(_sticky_step_opts.perceived_s);
  const frame = plot_series.sticky_step_frame(
    agent,
    _sticky_session.environment,
    _sticky_session.state,
    { day: _sticky_session.day, last_info: _sticky_session.last_info }
  );
  frame.meta.finished = _sticky_session.finished;
  return frame;
}

function jsonOk(body, status = 200) {
  return { ok: status >= 200 && status < 300, status, body };
}

function jsonError(error, fields = null, status = 400) {
  const body = { ok: false, error };
  if (fields) body.fields = fields;
  return { ok: false, status, body };
}

/** Mirror of POST /api/config */
export function apiConfig(data = {}) {
  const cfg = {
    ss_value: data.ss_value ?? _config.ss_value,
    ss_delay: data.ss_delay ?? _config.ss_delay,
    ll_value: data.ll_value ?? _config.ll_value,
    ll_delay: data.ll_delay ?? _config.ll_delay,
    length: "length" in data ? data.length : _config.length,
    horizon: "horizon" in data ? data.horizon : _config.horizon,
    vision: data.vision ?? _config.vision,
    discount_factor: data.discount_factor ?? _config.discount_factor,
    max_steps: "max_steps" in data ? data.max_steps : _config.max_steps,
  };
  if (cfg.vision !== "short" && cfg.vision !== "long") {
    return jsonError("vision must be short or long");
  }
  const err = validateConfig(cfg);
  if (err) {
    const fields = [];
    const low = err.toLowerCase();
    if (low.includes("length") || low.includes("horizon")) {
      fields.push("length", "horizon");
    }
    if (low.includes("max steps")) fields.push("max_steps");
    if (low.includes("discount")) fields.push("discount_factor");
    return jsonError(err, fields.length ? fields : ["length", "horizon"]);
  }
  try {
    cfg.length = parseOptionalInt(cfg.length);
    cfg.horizon = parseOptionalInt(cfg.horizon);
    cfg.max_steps = parseOptionalInt(cfg.max_steps);
    cfg.discount_factor = Number(cfg.discount_factor);
    rebuildSession(cfg);
  } catch (exc) {
    return jsonError(String(exc.message || exc), ["length", "horizon"]);
  }
  return jsonOk({ ok: true, config: structuredClone(_config) });
}

/** Mirror of POST /api/steppable/reset */
export function apiSteppableReset(demo = "visualizing_main") {
  if (demo === "hyperbolic_curve") {
    resetHypSession();
    return jsonOk(currentHypFrame());
  }
  if (demo === "hyperbolic_ss_ll") {
    resetHypSsLlSession();
    return jsonOk(currentHypSsLlFrame());
  }
  if (demo === "sticky_step") {
    rebuildStickySession();
    return jsonOk(currentStickyStepFrame());
  }
  rebuildSession();
  return jsonOk(currentSailsFrame());
}

/** Mirror of POST /api/steppable/step */
export function apiSteppableStep(demo = "visualizing_main") {
  if (demo === "hyperbolic_curve") {
    if (_hyp_session.finished) return jsonOk(currentHypFrame());
    const delay = Math.trunc(_config.ll_delay);
    if (_hyp_session.elapsed >= delay) {
      _hyp_session.finished = true;
      return jsonOk(currentHypFrame());
    }
    _hyp_session.elapsed += 1;
    if (_hyp_session.elapsed >= delay) _hyp_session.finished = true;
    return jsonOk(currentHypFrame());
  }

  if (demo === "hyperbolic_ss_ll") {
    if (_hyp_ss_ll_session.finished) return jsonOk(currentHypSsLlFrame());
    const delay = hypSsLlMaxDelay();
    if (_hyp_ss_ll_session.elapsed >= delay) {
      _hyp_ss_ll_session.finished = true;
      return jsonOk(currentHypSsLlFrame());
    }
    _hyp_ss_ll_session.elapsed += 1;
    if (_hyp_ss_ll_session.elapsed >= delay) _hyp_ss_ll_session.finished = true;
    return jsonOk(currentHypSsLlFrame());
  }

  if (demo === "sticky_step") {
    if (_sticky_session.environment == null) {
      rebuildStickySession();
      return jsonOk(currentStickyStepFrame());
    }
    if (_sticky_session.finished) return jsonOk(currentStickyStepFrame());

    const agent = _sticky_session.agent;
    agent.learn_stickiness = Boolean(_sticky_step_opts.learn_stickiness);
    agent.perceived_s = Number(_sticky_step_opts.perceived_s);
    const environment = _sticky_session.environment;
    let state = _sticky_session.state;
    const day = _sticky_session.day;

    const [intent] = agent.choose_action(state, environment);
    const stickiness = agent.perceived_stickiness(environment);
    const [nextState, finished, info] = environment.step(intent, stickiness);
    _sticky_session.state = nextState;
    _sticky_session.last_info = info;

    const cap = _config.max_steps;
    const hit_cap = cap != null && day + 1 >= cap;
    if (finished || hit_cap) {
      _sticky_session.finished = true;
      return jsonOk(currentStickyStepFrame());
    }
    _sticky_session.day = day + 1;
    return jsonOk(currentStickyStepFrame());
  }

  if (_session.environment == null) {
    rebuildSession();
    return jsonOk(currentSailsFrame());
  }
  if (_session.finished) return jsonOk(currentSailsFrame());

  const agent = _session.agent;
  const environment = _session.environment;
  let state = _session.state;
  const day = _session.day;

  const [action] = agent.choose_action(state, environment);
  const [nextState, finished] = environment.step(action);
  _session.state = nextState;

  const cap = _config.max_steps;
  const hit_cap = cap != null && day + 1 >= cap;
  if (finished || hit_cap) {
    _session.finished = true;
    return jsonOk(currentSailsFrame());
  }
  _session.day = day + 1;
  return jsonOk(currentSailsFrame());
}

/** Mirror of POST /api/steppable/step_back */
export function apiSteppableStepBack(demo = "visualizing_main") {
  if (demo !== "hyperbolic_ss_ll") {
    return jsonError("step_back only for hyperbolic_ss_ll");
  }
  if (_hyp_ss_ll_session.elapsed > 0) _hyp_ss_ll_session.elapsed -= 1;
  _hyp_ss_ll_session.finished = false;
  return jsonOk(currentHypSsLlFrame());
}

/** Mirror of GET /api/steppable/frame */
export function apiSteppableFrame(demo = "visualizing_main") {
  if (demo === "hyperbolic_curve") return jsonOk(currentHypFrame());
  if (demo === "hyperbolic_ss_ll") return jsonOk(currentHypSsLlFrame());
  if (demo === "sticky_step") return jsonOk(currentStickyStepFrame());
  if (_session.environment == null) rebuildSession();
  return jsonOk(currentSailsFrame());
}

/** Mirror of POST /api/steppable/sticky_options */
export function apiStickyOptions(data = {}) {
  let rebuild_needed = false;

  if ("learn_stickiness" in data) {
    _sticky_step_opts.learn_stickiness = Boolean(data.learn_stickiness);
  }

  for (const key of ["prior_alpha", "prior_beta"]) {
    if (!(key in data)) continue;
    const val = Number(data[key]);
    if (!Number.isFinite(val)) return jsonError(`${key} must be a number`);
    if (val <= 0.0) return jsonError(`${key} must be positive`);
    if (Math.abs(val - Number(_sticky_step_opts[key])) > 1e-12) {
      _sticky_step_opts[key] = val;
      rebuild_needed = true;
    }
  }

  if ("seed" in data) {
    let seed;
    try {
      seed = parseOptionalInt(data.seed);
    } catch {
      return jsonError("seed must be an integer or blank");
    }
    if (seed != null && !Number.isInteger(Number(data.seed)) && String(data.seed).trim() !== "") {
      // allow numeric ints only
    }
    if (seed !== _sticky_step_opts.seed) {
      _sticky_step_opts.seed = seed;
      rebuild_needed = true;
    }
  }

  if (rebuild_needed || _sticky_session.environment == null) {
    rebuildStickySession();
  } else {
    syncStickyPresetFromPrior();
    const agent = _sticky_session.agent;
    agent.learn_stickiness = Boolean(_sticky_step_opts.learn_stickiness);
    agent.perceived_s = Number(_sticky_step_opts.perceived_s);
  }

  const frame = currentStickyStepFrame();
  frame.ok = true;
  frame.options = structuredClone(_sticky_step_opts);
  return jsonOk(frame);
}

/** Mirror of GET /api/demo/<name> */
export function apiDemo(name, { stickiness = DEFAULT_STICKINESS } = {}) {
  const [environment, agent, state] = freshSnapshot();
  if (name === "visualise_sails") {
    return jsonOk(plot_series.sails_frame(agent, environment, state));
  }
  if (name === "visualise_sticky_sails") {
    let s = Number(stickiness);
    if (!Number.isFinite(s)) {
      return jsonError("s must be a number in [0, 1]");
    }
    s = Math.max(0.0, Math.min(1.0, s));
    const sticky_agent = new StickyHypAgent({
      ...buildAgentParams(_config),
      perceived_s: s,
    });
    return jsonOk(plot_series.sticky_sails_frame(sticky_agent, environment, state));
  }
  if (name === "aggregate_sail") {
    return jsonOk(plot_series.aggregate_frame(agent, environment, state));
  }
  return jsonError(`Unknown demo: ${name}`, null, 404);
}

/** Mirror of GET /api/sticky */
export function apiSticky({
  delay_max = DEFAULT_DELAY_MAX,
  delay_step = DEFAULT_DELAY_STEP,
} = {}) {
  const [environment, agent, state] = freshSnapshot();
  return jsonOk(
    plot_series.sticky_curves(
      agent,
      environment,
      state,
      delay_max,
      delay_step,
      _config.ll_value
    )
  );
}

/** Fetch-compatible helper used by app.js */
export async function apiFetch(path, options = {}) {
  const method = (options.method || "GET").toUpperCase();
  let data = null;
  if (options.body) {
    data = typeof options.body === "string" ? JSON.parse(options.body) : options.body;
  }

  const url = new URL(path, "http://local/");
  const pathname = url.pathname;
  const demo = url.searchParams.get("demo") || (data && data.demo) || "visualizing_main";

  let result;
  if (pathname === "/api/config" && method === "POST") {
    result = apiConfig(data || {});
  } else if (pathname === "/api/steppable/reset" && method === "POST") {
    result = apiSteppableReset(demo);
  } else if (pathname === "/api/steppable/step" && method === "POST") {
    result = apiSteppableStep(demo);
  } else if (pathname === "/api/steppable/step_back" && method === "POST") {
    result = apiSteppableStepBack(demo);
  } else if (pathname === "/api/steppable/frame") {
    result = apiSteppableFrame(demo);
  } else if (pathname === "/api/steppable/sticky_options" && method === "POST") {
    result = apiStickyOptions(data || {});
  } else if (pathname.startsWith("/api/demo/")) {
    const name = pathname.slice("/api/demo/".length);
    const stickiness = url.searchParams.has("stickiness")
      ? Number(url.searchParams.get("stickiness"))
      : DEFAULT_STICKINESS;
    result = apiDemo(name, { stickiness });
  } else if (pathname === "/api/sticky") {
    result = apiSticky({
      delay_max: Number(url.searchParams.get("delay_max") ?? DEFAULT_DELAY_MAX),
      delay_step: Number(url.searchParams.get("delay_step") ?? DEFAULT_DELAY_STEP),
    });
  } else {
    result = jsonError(`Unknown path: ${pathname}`, null, 404);
  }

  return {
    ok: result.ok,
    status: result.status,
    async json() {
      return result.body;
    },
  };
}

// Initialise once (same as demo_web_app rebuild_session at import)
rebuildSession();
