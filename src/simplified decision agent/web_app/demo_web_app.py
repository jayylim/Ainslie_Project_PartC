"""
Unified interactive web app for the visual demos.

Shared top-level params (rewards, delays, env length, horizon, vision).
Three sections:
  - Steppable: visualizing_main (env days) | hyperbolic_curve (reward approaching day 0)
  - Non-steppable: visualise_sails | aggregate_sail
  - Sticky: sticky_graph + visualise_sticky_sails (stacked; click current curve sets s)

Math lives in Python (core agent/env + plot_series); the browser draws with Plotly.

Run from the package root:

    python -m web_app.demo_web_app

then open http://127.0.0.1:5050

(Port 5050 avoids macOS AirPlay, which often claims :5000 and makes
Firefox fail on http://localhost:5000 with 403.)

For hosting, bind 0.0.0.0 and honour $PORT (see render.yaml).
"""

import copy
import sys
from pathlib import Path

_WEB_DIR = Path(__file__).resolve().parent
_PKG_ROOT = _WEB_DIR.parent
sys.path.insert(0, str(_PKG_ROOT))
sys.path.insert(0, str(_WEB_DIR))

from flask import Flask, jsonify, render_template_string, request

from agents.hyperbolic import HypAgent
from agents.sticky import StickyHypAgent
from environments.timeline import TimelineEnvironment
from main import (
    agent_params,
    env_params,
    max_steps as default_max_steps,
    stickiness as default_stickiness,
)
import plot_series

# === Defaults (seeded from main.py) ===
DEFAULT_DELAY_MAX = 24
DEFAULT_DELAY_STEP = 4
DEFAULT_STICKINESS = float(default_stickiness)

app = Flask(__name__)

# Defaults from main.py (immutable seed for "Reset to default")
_DEFAULT_CONFIG = {
    "ss_value": env_params["rewards"]["SS"]["value"],
    "ss_delay": env_params["rewards"]["SS"]["delay"],
    "ll_value": env_params["rewards"]["LL"]["value"],
    "ll_delay": env_params["rewards"]["LL"]["delay"],
    "length": env_params["length"],
    "horizon": agent_params["horizon"],
    "vision": agent_params["vision"],
    "discount_factor": agent_params["discount_factor"],
    # Same semantics as main.max_steps / visualizing_main: None = no step cap
    "max_steps": default_max_steps,
}

# Mutable live config + steppable session (single local user)
_config = copy.deepcopy(_DEFAULT_CONFIG)
_decision_interval = env_params["decision_interval"]
_start = env_params["start"]

_session = {
    "environment": None,
    "agent": None,
    "state": None,
    "day": 0,
    "finished": False,
}

# Hyperbolic approach: agent fixed at day 0; elapsed hours toward LL delivery
_hyp_session = {
    "elapsed": 0,
    "finished": False,
}


def _parse_optional_int(value):
    """Blank / None → None; otherwise int. Rejects non-numeric strings."""
    if value is None or value == "":
        return None
    return int(value)


def _build_env_params(cfg):
    return {
        "length": _parse_optional_int(cfg["length"]),
        "start": _start,
        "decision_interval": _decision_interval,
        "rewards": {
            "SS": {"value": float(cfg["ss_value"]), "delay": int(cfg["ss_delay"])},
            "LL": {"value": float(cfg["ll_value"]), "delay": int(cfg["ll_delay"])},
        },
    }


def _build_agent_params(cfg):
    horizon = _parse_optional_int(cfg["horizon"])
    if horizon == 0:
        raise ValueError("Agent horizon cannot be 0 (use blank for none)")
    return {
        "discount_factor": float(cfg["discount_factor"]),
        "vision": cfg["vision"],
        "horizon": horizon,
    }


def validate_config(cfg):
    """
    Return an error message if the config is invalid, else None.

    Rules aligned with core / visualizing_main:
      - Infinite env (length none) requires a finite agent horizon.
      - max_steps may be none (no run cap); if set, must be ≥ 1.
      - discount_factor must be a finite number ≥ 0.
    """
    length = _parse_optional_int(cfg.get("length"))
    horizon = _parse_optional_int(cfg.get("horizon"))
    max_steps_val = _parse_optional_int(cfg.get("max_steps"))
    if length is None and horizon is None:
        return (
            "Env length and agent horizon cannot both be none — "
            "set at least one (infinite timeline needs a finite horizon)."
        )
    if horizon == 0:
        return "Agent horizon cannot be 0 (leave blank for none, or use ≥ 1)."
    if max_steps_val == 0:
        return "Max steps cannot be 0 (leave blank for none, or use ≥ 1)."
    try:
        k = float(cfg["discount_factor"])
    except (TypeError, ValueError):
        return "Discount factor must be a number."
    if k < 0 or k != k:  # NaN check
        return "Discount factor must be ≥ 0."
    return None


def rebuild_session(cfg=None):
    """Rebuild agent/env from config and reset both steppable demos."""
    if cfg is not None:
        _config.update(cfg)
    environment = TimelineEnvironment(**_build_env_params(_config))
    agent = HypAgent(**_build_agent_params(_config))
    state = environment.reset()
    _session.update(
        {
            "environment": environment,
            "agent": agent,
            "state": state,
            "day": 1,
            "finished": False,
        }
    )
    reset_hyp_session()
    return _session


def reset_hyp_session():
    """Restart hyperbolic approach at elapsed = 0 (full remaining delay)."""
    _hyp_session.update({"elapsed": 0, "finished": False})
    return _hyp_session


def fresh_snapshot():
    """Independent env/agent/state at t=0 for non-steppable / sticky demos."""
    environment = TimelineEnvironment(**_build_env_params(_config))
    agent = HypAgent(**_build_agent_params(_config))
    state = environment.reset()
    return environment, agent, state


def current_sails_frame():
    """visualizing_main sails frame for the current env day."""
    frame = plot_series.sails_frame(
        _session["agent"],
        _session["environment"],
        _session["state"],
        day=_session["day"],
    )
    frame["meta"]["finished"] = _session["finished"]
    return frame


def current_hyp_frame():
    """hyperbolic_curve frame: reward approaching day 0."""
    frame = plot_series.hyperbolic_frame(
        _config["ll_value"],
        _config["ll_delay"],
        _config["discount_factor"],
        elapsed=_hyp_session["elapsed"],
    )
    frame["meta"]["finished"] = _hyp_session["finished"]
    return frame


def _steppable_demo():
    """Active steppable demo from query or JSON body (default visualizing_main)."""
    demo = request.args.get("demo")
    if not demo and request.method in ("POST", "PUT", "PATCH"):
        data = request.get_json(force=True, silent=True) or {}
        demo = data.get("demo")
    return demo or "visualizing_main"


# Initialise once at import
rebuild_session()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/api/config", methods=["POST"])
def api_config():
    data = request.get_json(force=True, silent=True) or {}
    # Explicit null when client sends null for optional ints
    cfg = {
        "ss_value": data.get("ss_value", _config["ss_value"]),
        "ss_delay": data.get("ss_delay", _config["ss_delay"]),
        "ll_value": data.get("ll_value", _config["ll_value"]),
        "ll_delay": data.get("ll_delay", _config["ll_delay"]),
        "length": data["length"] if "length" in data else _config["length"],
        "horizon": data["horizon"] if "horizon" in data else _config["horizon"],
        "vision": data.get("vision", _config["vision"]),
        "discount_factor": data.get("discount_factor", _config["discount_factor"]),
        "max_steps": data["max_steps"] if "max_steps" in data else _config["max_steps"],
    }
    if cfg["vision"] not in ("short", "long"):
        return jsonify({"ok": False, "error": "vision must be short or long"}), 400
    err = validate_config(cfg)
    if err:
        fields = []
        low = err.lower()
        if "length" in low or "horizon" in low:
            fields.extend(["length", "horizon"])
        if "max steps" in low:
            fields.append("max_steps")
        if "discount" in low:
            fields.append("discount_factor")
        return jsonify({"ok": False, "error": err, "fields": fields or ["length", "horizon"]}), 400
    try:
        # Store parsed optionals so session reads clean Nones
        cfg["length"] = _parse_optional_int(cfg["length"])
        cfg["horizon"] = _parse_optional_int(cfg["horizon"])
        cfg["max_steps"] = _parse_optional_int(cfg["max_steps"])
        cfg["discount_factor"] = float(cfg["discount_factor"])
        rebuild_session(cfg)
    except (ValueError, TypeError) as exc:
        return jsonify({"ok": False, "error": str(exc), "fields": ["length", "horizon"]}), 400
    return jsonify({"ok": True, "config": copy.deepcopy(_config)})


@app.route("/api/steppable/reset", methods=["POST"])
def api_steppable_reset():
    demo = _steppable_demo()
    if demo == "hyperbolic_curve":
        reset_hyp_session()
        return jsonify(current_hyp_frame())
    rebuild_session()
    return jsonify(current_sails_frame())


@app.route("/api/steppable/step", methods=["POST"])
def api_steppable_step():
    """
    visualizing_main: step the env one day (same as matplotlib click loop).
    hyperbolic_curve: advance elapsed by 1h toward delivery (agent stays at day 0).
    """
    demo = _steppable_demo()

    if demo == "hyperbolic_curve":
        if _hyp_session["finished"]:
            return jsonify(current_hyp_frame())
        delay = int(_config["ll_delay"])
        # After showing remaining=0, next step marks finished (delay+1 frames total)
        if _hyp_session["elapsed"] >= delay:
            _hyp_session["finished"] = True
            return jsonify(current_hyp_frame())
        _hyp_session["elapsed"] += 1
        if _hyp_session["elapsed"] >= delay:
            _hyp_session["finished"] = True
        return jsonify(current_hyp_frame())

    if _session["environment"] is None:
        rebuild_session()
        return jsonify(current_sails_frame())

    if _session["finished"]:
        return jsonify(current_sails_frame())

    agent = _session["agent"]
    environment = _session["environment"]
    state = _session["state"]
    day = _session["day"]

    action, _ = agent.choose_action(state, environment)
    state, finished = environment.step(action)
    _session["state"] = state

    # visualizing_main: while not finished and (max_steps is None or steps < max_steps)
    cap = _config.get("max_steps")
    hit_cap = cap is not None and day >= cap
    if finished or hit_cap:
        _session["finished"] = True
        return jsonify(current_sails_frame())

    _session["day"] = day + 1
    return jsonify(current_sails_frame())


@app.route("/api/steppable/frame")
def api_steppable_frame():
    demo = _steppable_demo()
    if demo == "hyperbolic_curve":
        return jsonify(current_hyp_frame())
    if _session["environment"] is None:
        rebuild_session()
    return jsonify(current_sails_frame())


@app.route("/api/demo/<name>")
def api_demo(name):
    environment, agent, state = fresh_snapshot()
    if name == "visualise_sails":
        return jsonify(plot_series.sails_frame(agent, environment, state))
    if name == "visualise_sticky_sails":
        s = request.args.get("stickiness", DEFAULT_STICKINESS, type=float)
        if s is None or s != s:  # None / NaN
            return jsonify({"error": "stickiness must be a number in [0, 1]"}), 400
        s = max(0.0, min(1.0, float(s)))
        sticky_agent = StickyHypAgent(
            **_build_agent_params(_config), stickiness=s
        )
        return jsonify(
            plot_series.sticky_sails_frame(sticky_agent, environment, state)
        )
    if name == "aggregate_sail":
        return jsonify(plot_series.aggregate_frame(agent, environment, state))
    return jsonify({"error": f"Unknown demo: {name}"}), 404


@app.route("/api/sticky")
def api_sticky():
    delay_max = request.args.get("delay_max", DEFAULT_DELAY_MAX, type=float)
    delay_step = request.args.get("delay_step", DEFAULT_DELAY_STEP, type=float)
    environment, agent, state = fresh_snapshot()
    return jsonify(
        plot_series.sticky_curves(
            agent,
            environment,
            state,
            delay_max,
            delay_step,
            reward_LL=_config["ll_value"],
        )
    )


@app.route("/")
def index():
    # Form fields seed from live config; DEFAULT_CONFIG in JS is always main.py seed
    return render_template_string(
        PAGE,
        ss_value=_config["ss_value"],
        ss_delay=_config["ss_delay"],
        ll_value=_config["ll_value"],
        ll_delay=_config["ll_delay"],
        length=_config["length"] if _config["length"] is not None else "",
        horizon=_config["horizon"] if _config["horizon"] is not None else "",
        vision=_config["vision"],
        discount_factor=_config["discount_factor"],
        max_steps=_config["max_steps"] if _config["max_steps"] is not None else "",
        default_ss_value=_DEFAULT_CONFIG["ss_value"],
        default_ss_delay=_DEFAULT_CONFIG["ss_delay"],
        default_ll_value=_DEFAULT_CONFIG["ll_value"],
        default_ll_delay=_DEFAULT_CONFIG["ll_delay"],
        default_length=_DEFAULT_CONFIG["length"],
        default_horizon=_DEFAULT_CONFIG["horizon"],
        default_vision=_DEFAULT_CONFIG["vision"],
        default_discount_factor=_DEFAULT_CONFIG["discount_factor"],
        default_max_steps=_DEFAULT_CONFIG["max_steps"],
        default_delay_max=DEFAULT_DELAY_MAX,
        default_delay_step=DEFAULT_DELAY_STEP,
        default_stickiness=DEFAULT_STICKINESS,
    )


PAGE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Decision Agent — Visual Demos</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  :root {
    --bg: #f6f7f9;
    --panel: #ffffff;
    --ink: #1a1d23;
    --muted: #5c6570;
    --line: #d8dde5;
    --accent: #1f4e79;
    --accent-soft: #e4edf5;
    --resist: #1f4e79;
    --snack: #c44e52;
    --error: #b42318;
    --error-bg: #fef3f2;
    --error-border: #fecdca;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
    background: linear-gradient(165deg, #eef2f6 0%, #f7f5f1 45%, #e9eef3 100%);
    color: var(--ink);
    min-height: 100vh;
  }
  .wrap {
    max-width: min(1680px, 90vw);
    margin: 0 auto;
    padding: 1.25rem 1.25rem 2.5rem;
  }
  h1 {
    font-size: 1.55rem;
    font-weight: 600;
    letter-spacing: -0.02em;
    margin: 0 0 0.25rem;
  }
  .subtitle {
    color: var(--muted);
    font-size: 0.95rem;
    margin: 0 0 1.1rem;
  }
  .panel {
    background: var(--panel);
    border: 1px solid #d8dde5;
    border-radius: 10px;
    padding: 0.9rem 1rem;
    margin-bottom: 0.85rem;
    box-shadow: 0 1px 2px rgba(26, 29, 35, 0.04);
  }
  .panel-label {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.75rem;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--muted);
    margin-bottom: 0.65rem;
    font-family: ui-sans-serif, system-ui, sans-serif;
  }
  .panel-label button.secondary {
    text-transform: none;
    letter-spacing: 0;
    background: #fff;
    color: var(--ink);
    border: 1px solid #c9d0da;
    padding: 0.35rem 0.75rem;
    border-radius: 6px;
    cursor: pointer;
    font-size: 0.82rem;
  }
  .params {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
    gap: 0.75rem 1rem;
    align-items: end;
  }
  label {
    display: flex;
    flex-direction: column;
    gap: 0.28rem;
    font-size: 0.8rem;
    color: var(--muted);
    font-family: ui-sans-serif, system-ui, sans-serif;
  }
  input[type="number"], select {
    font: inherit;
    font-size: 0.95rem;
    color: var(--ink);
    padding: 0.4rem 0.5rem;
    border: 1px solid #c9d0da;
    border-radius: 6px;
    background: #fafbfc;
  }
  input.field-error {
    border-color: var(--error);
    background: var(--error-bg);
    outline: 2px solid var(--error-border);
  }
  .param-hint {
    grid-column: 1 / -1;
    font-size: 0.78rem;
    color: var(--muted);
    font-family: ui-sans-serif, system-ui, sans-serif;
    margin: 0;
  }
  #configError {
    display: none;
    grid-column: 1 / -1;
    font-size: 0.85rem;
    font-family: ui-sans-serif, system-ui, sans-serif;
    color: var(--error);
    background: var(--error-bg);
    border: 1px solid var(--error-border);
    border-radius: 6px;
    padding: 0.55rem 0.7rem;
  }
  #configError.visible { display: block; }
  input[type="range"] { width: 100%; accent-color: var(--accent); }
  .vision-toggle {
    display: flex;
    gap: 0.35rem;
    font-family: ui-sans-serif, system-ui, sans-serif;
  }
  .vision-toggle button {
    flex: 1;
    border: 1px solid #c9d0da;
    background: #fafbfc;
    color: var(--muted);
    padding: 0.4rem 0.6rem;
    border-radius: 6px;
    cursor: pointer;
    font-size: 0.9rem;
  }
  .vision-toggle button.active {
    background: var(--accent-soft);
    border-color: var(--accent);
    color: var(--accent);
    font-weight: 600;
  }
  .tabs {
    display: flex;
    gap: 0.4rem;
    margin-bottom: 0.85rem;
    font-family: ui-sans-serif, system-ui, sans-serif;
  }
  .tabs button {
    border: 1px solid #c9d0da;
    background: #fafbfc;
    color: var(--muted);
    padding: 0.55rem 1rem;
    border-radius: 8px;
    cursor: pointer;
    font-size: 0.92rem;
  }
  .tabs button.active {
    background: var(--ink);
    border-color: var(--ink);
    color: #fff;
  }
  .chrome {
    display: flex;
    flex-wrap: wrap;
    gap: 0.75rem 1.25rem;
    align-items: center;
    margin-bottom: 0.5rem;
    font-family: ui-sans-serif, system-ui, sans-serif;
    font-size: 0.88rem;
  }
  .chrome.hidden { display: none; }
  .chrome button.action {
    background: var(--accent);
    color: #fff;
    border: none;
    padding: 0.45rem 0.95rem;
    border-radius: 6px;
    cursor: pointer;
    font-size: 0.9rem;
  }
  .chrome button.action:disabled {
    opacity: 0.45;
    cursor: not-allowed;
  }
  .chrome button.secondary {
    background: #fff;
    color: var(--ink);
    border: 1px solid #c9d0da;
    padding: 0.45rem 0.95rem;
    border-radius: 6px;
    cursor: pointer;
  }
  .range-pair {
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
    min-width: 160px;
  }
  .range-pair span { color: var(--muted); font-size: 0.8rem; }
  .meta-line {
    color: var(--muted);
    font-size: 0.85rem;
    font-family: ui-sans-serif, system-ui, sans-serif;
    min-height: 1.2em;
    margin: 0.35rem 0 0.5rem;
  }
  .meta-line .choice {
    color: var(--ink);
    font-weight: 800;
    opacity: 1;
  }
  .chart-box {
    background: #fff;
    border: 1px solid #d8dde5;
    border-radius: 10px;
    padding: 0.4rem;
  }
  #stickyStack {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
  }
  #stickyStack.hidden { display: none; }
  #singleView.hidden { display: none; }
  .sticky-plot-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--muted);
    font-family: ui-sans-serif, system-ui, sans-serif;
    margin: 0 0 0.2rem;
  }
</style>
</head>
<body>
<div class="wrap">
  <h1>Decision Agent demos</h1>
  <p class="subtitle">Shared parameters · switch between steppable, static, and sticky views</p>

  <div class="panel">
    <div class="panel-label">
      <span>Main parameters</span>
      <button type="button" class="secondary" id="btnResetDefaults">Reset to default</button>
    </div>
    <div class="params">
      <label>SS value
        <input type="number" id="ssValue" value="{{ ss_value }}" min="0" step="0.5"/>
      </label>
      <label>SS delay (h)
        <input type="number" id="ssDelay" value="{{ ss_delay }}" min="0" step="1"/>
      </label>
      <label>LL value
        <input type="number" id="llValue" value="{{ ll_value }}" min="0" step="0.5"/>
      </label>
      <label>LL delay (h)
        <input type="number" id="llDelay" value="{{ ll_delay }}" min="0" step="1"/>
      </label>
      <label>Env length (h)
        <input type="number" id="length" value="{{ length }}" min="1" step="24" placeholder="blank = none"/>
      </label>
      <label>Agent horizon
        <input type="number" id="horizon" value="{{ horizon }}" min="1" step="1" placeholder="blank = none"/>
      </label>
      <label>Discount factor (k)
        <input type="number" id="discountFactor" value="{{ discount_factor }}" min="0" step="0.05"/>
      </label>
      <label>Max steps
        <input type="number" id="maxSteps" value="{{ max_steps }}" min="1" step="1" placeholder="blank = none"/>
      </label>
      <label>Vision
        <div class="vision-toggle">
          <button type="button" id="visionShort" data-v="short">Short</button>
          <button type="button" id="visionLong" data-v="long">Long</button>
        </div>
      </label>
      <p class="param-hint">
        Clear Env length or Agent horizon for <em>none</em> (infinite env, or plan all remaining days) —
        at least one must be set. Max steps blank = no run cap (same as <code>main.max_steps = None</code>);
        with a finite env the timeline can still end first.
      </p>
      <div id="configError" role="alert"></div>
    </div>
  </div>

  <div class="tabs">
    <button type="button" data-section="steppable" class="active">Steppable</button>
    <button type="button" data-section="nonsteppable">Non-steppable</button>
    <button type="button" data-section="sticky">Sticky</button>
  </div>

  <div class="panel">
    <div id="chromeSteppable" class="chrome">
      <label style="flex-direction:row;align-items:center;gap:0.5rem;">
        Demo
        <select id="steppableDemoSelect">
          <option value="visualizing_main">visualizing_main</option>
          <option value="hyperbolic_curve">hyperbolic_curve</option>
        </select>
      </label>
      <button type="button" class="secondary" id="btnReset">Reset</button>
      <button type="button" class="action" id="btnStep">Step</button>
      <span id="stepHint">Step advances the selected demo</span>
    </div>
    <div id="chromeNon" class="chrome hidden">
      <label style="flex-direction:row;align-items:center;gap:0.5rem;">
        Demo
        <select id="demoSelect">
          <option value="visualise_sails">visualise_sails</option>
          <option value="aggregate_sail">aggregate_sail</option>
        </select>
      </label>
    </div>
    <div id="chromeSticky" class="chrome hidden">
      <div class="range-pair">
        <span>LL delay max: <strong id="delayMaxVal">{{ default_delay_max }}h</strong></span>
        <input type="range" id="delayMax" min="0" max="48" step="1" value="{{ default_delay_max }}"/>
      </div>
      <div class="range-pair">
        <span>LL delay step: <strong id="delayStepVal">{{ default_delay_step }}h</strong></span>
        <input type="range" id="delayStep" min="1" max="12" step="1" value="{{ default_delay_step }}"/>
      </div>
      <label style="min-width:120px;">Stickiness s
        <input type="number" id="stickiness" value="{{ default_stickiness }}" min="0" max="1" step="0.01"/>
      </label>
    </div>

    <div id="singleView">
      <div id="meta" class="meta-line"></div>
      <div id="chart" class="chart-box"></div>
    </div>
    <div id="stickyStack" class="hidden">
      <div>
        <p class="sticky-plot-label">Sticky graph · click black (current) curve to set s</p>
        <div id="metaGraph" class="meta-line"></div>
        <div id="chartGraph" class="chart-box"></div>
      </div>
      <div>
        <p class="sticky-plot-label">Sticky sails at s</p>
        <div id="metaSails" class="meta-line"></div>
        <div id="chartSails" class="chart-box"></div>
      </div>
    </div>
  </div>
</div>

<script>
const DEFAULT_CONFIG = {
  ss_value: {{ default_ss_value|tojson }},
  ss_delay: {{ default_ss_delay|tojson }},
  ll_value: {{ default_ll_value|tojson }},
  ll_delay: {{ default_ll_delay|tojson }},
  length: {{ default_length|tojson }},
  horizon: {{ default_horizon|tojson }},
  discount_factor: {{ default_discount_factor|tojson }},
  max_steps: {{ default_max_steps|tojson }},
  vision: {{ default_vision|tojson }},
};
const DEFAULT_VISION = DEFAULT_CONFIG.vision;
const DEFAULT_DELAY_MAX = {{ default_delay_max }};
const DEFAULT_DELAY_STEP = {{ default_delay_step }};
const DEFAULT_STICKINESS = {{ default_stickiness|tojson }};

let vision = DEFAULT_VISION;
let section = "steppable";
let stickyClickBound = false;
let configTimer = null;
let finished = false;

const metaEl = document.getElementById("meta");
const metaGraphEl = document.getElementById("metaGraph");
const metaSailsEl = document.getElementById("metaSails");
const singleViewEl = document.getElementById("singleView");
const stickyStackEl = document.getElementById("stickyStack");
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
const demoSelect = document.getElementById("demoSelect");
const steppableDemoSelect = document.getElementById("steppableDemoSelect");
const delayMaxEl = document.getElementById("delayMax");
const delayStepEl = document.getElementById("delayStep");
const stickinessEl = document.getElementById("stickiness");
const btnStep = document.getElementById("btnStep");
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
  syncVisionButtons();
  clearConfigError();
}

function steppableDemo() {
  return steppableDemoSelect.value || "visualizing_main";
}

function syncStepHint() {
  if (steppableDemo() === "hyperbolic_curve") {
    stepHint.textContent =
      "hyperbolic_curve — Step moves the reward one hour closer to day 0 (axes stay fixed)";
  } else {
    stepHint.textContent =
      "visualizing_main — Step advances one decision day in the environment";
  }
}
syncStepHint();

function syncVisionButtons() {
  visionShort.classList.toggle("active", vision === "short");
  visionLong.classList.toggle("active", vision === "long");
}
syncVisionButtons();

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
  const res = await fetch("/api/config", {
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
      font: { family: "Iowan Old Style, Palatino Linotype, Georgia, serif", size: 13 },
      paper_bgcolor: "#fff",
      plot_bgcolor: "#fafbfc",
    },
    payload.layout || {}
  );
  if (!layout.uirevision) {
    layout.uirevision = uirevision;
  }
  return layout;
}

function fillMeta(el, payload) {
  const m = payload.meta || {};
  if (payload.kind === "sticky") {
    el.textContent =
      `delays = [${(m.delays || []).join(", ")}]h · LL = ${m.reward_LL} · ` +
      `horizon = ${m.horizon} · discount_factor = ${m.discount_factor}`;
  } else if (m.action_values) {
    const av = m.action_values;
    const dayBit = m.day != null ? `Day ${m.day} · ` : "";
    const stickyBit =
      m.stickiness != null ? `s = ${Number(m.stickiness).toFixed(2)} · ` : "";
    const doneBit = m.finished ? " · finished" : "";
    el.innerHTML =
      dayBit +
      stickyBit +
      `<span class="choice">Agent chooses ${m.choice}</span>` +
      ` · snack = ${av.snack.toFixed(4)} · resist = ${av.resist.toFixed(4)}` +
      doneBit;
  } else if (payload.kind === "hyperbolic_curve") {
    const doneBit = m.finished ? " · finished" : "";
    el.innerHTML =
      `elapsed = ${m.elapsed}h · remaining = ${m.remaining}h · ` +
      `<span class="choice">V(day 0) = ${m.perceived.toFixed(4)}</span>` +
      ` · k = ${m.discount_factor}` +
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
}

function ensureStickyClickHandler() {
  if (stickyClickBound) return;
  const chart = document.getElementById("chartGraph");
  chart.on("plotly_click", (ev) => {
    if (section !== "sticky") return;
    const pt = ev.points && ev.points[0];
    if (!pt) return;
    const isCurrent =
      (pt.data && pt.data.meta && pt.data.meta.is_current) ||
      String((pt.data && pt.data.name) || "").includes("(current)");
    if (!isCurrent) return;
    setStickiness(pt.x);
    refreshStickySailsOnly();
  });
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
  if (!plotReady[divId]) {
    Plotly.newPlot(divId, payload.traces, layout, opts);
    plotReady[divId] = true;
  } else {
    Plotly.react(divId, payload.traces, layout, opts);
  }
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
    drawInto("chartGraph", graphPayload);
    drawInto("chartSails", sailsPayload);
    ensureStickyClickHandler();
    fillMeta(metaGraphEl, graphPayload);
    fillMeta(metaSailsEl, sailsPayload);
  } catch (err) {
    metaGraphEl.textContent = String(err.message || err);
  }
}

async function fetchJson(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error("HTTP " + res.status);
  return res.json();
}

async function refreshStickySailsOnly() {
  try {
    const s = clampStickiness(stickinessEl.value);
    setStickiness(s);
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
      const s = clampStickiness(stickinessEl.value);
      setStickiness(s);
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
    const res = await fetch("/api/steppable/reset", {
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
  setStickiness(stickinessEl.value);
  if (section === "sticky") refreshStickySailsOnly();
});
syncStickyLabels();
syncViewPanes();

btnReset.addEventListener("click", async () => {
  try {
    await pushConfig();
    const res = await fetch("/api/steppable/reset", {
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

btnResetDefaults.addEventListener("click", () => {
  clearTimeout(configTimer);
  applyDefaultsToForm();
  setStickiness(DEFAULT_STICKINESS);
  applyConfigAndRefresh();
});

btnStep.addEventListener("click", async () => {
  try {
    const res = await fetch("/api/steppable/step", {
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

refresh();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    import os
    import webbrowser

    # Local default 5050: macOS AirPlay often binds :5000.
    # Cloud hosts (Render, etc.) inject PORT and require 0.0.0.0.
    port = int(os.environ.get("PORT", "5050"))
    host = os.environ.get("HOST") or (
        "0.0.0.0" if "PORT" in os.environ else "127.0.0.1"
    )
    print(f"Serving unified visual demos on {host}:{port}")
    if host in ("127.0.0.1", "localhost"):
        print("Open http://127.0.0.1:%s (use 127.0.0.1, not localhost)." % port)
        try:
            webbrowser.open(f"http://127.0.0.1:{port}")
        except Exception:
            pass
    app.run(debug=False, host=host, port=port, use_reloader=False)
