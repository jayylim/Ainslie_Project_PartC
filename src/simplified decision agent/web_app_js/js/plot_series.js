/** Port of web_app/plot_series.py — Plotly JSON frame builders. */

import { stick_p_t } from "./sticky_decisions.js";

export const COLORS = {
  dessert: "#c44e52",
  resist: "#4c72b0",
  dessert_sail: "#c44e52",
  resist_sail: "#4c72b0",
  belief: "#2ca02c",
  true_s: "#111111",
  perceived_s: "#ff7f0e",
};

export function fade(base_hex, i, n, max_fade = 0.72) {
  const h = base_hex.replace("#", "");
  let r = parseInt(h.slice(0, 2), 16);
  let g = parseInt(h.slice(2, 4), 16);
  let b = parseInt(h.slice(4, 6), 16);
  const t = (n > 1 ? i / (n - 1) : 0.0) * max_fade;
  r = Math.round(r + (255 - r) * t);
  g = Math.round(g + (255 - g) * t);
  b = Math.round(b + (255 - b) * t);
  return `#${r.toString(16).padStart(2, "0")}${g.toString(16).padStart(2, "0")}${b.toString(16).padStart(2, "0")}`;
}

export const LABEL_DISCOUNT_OFFSET = 2;
export const LABEL_AGG_OFFSET = 5;
export const SAIL_SAMPLE_POINTS = 200;

function _dense_sail_xy(agent, reward, delay, n_points = SAIL_SAMPLE_POINTS) {
  delay = Number(delay);
  if (delay <= 0) return [[0.0], [Number(agent.value_function(reward, 0))]];
  const n = Math.max(Number(n_points), 2);
  const xs = [];
  const ys = [];
  for (let i = 0; i < n; i++) {
    const t = (delay * i) / (n - 1);
    xs.push(t);
    ys.push(Number(agent.value_function(reward, delay - t)));
  }
  return [xs, ys];
}

function _dense_hyperbolic_xy(reward, remaining, k, n_points = SAIL_SAMPLE_POINTS) {
  remaining = Number(remaining);
  if (remaining <= 0) return [[0.0], [Number(reward)]];
  const n = Math.max(Number(n_points), 2);
  const xs = [];
  const ys = [];
  for (let i = 0; i < n; i++) {
    const t = (remaining * i) / (n - 1);
    xs.push(t);
    ys.push(Number(reward / (1.0 + k * (remaining - t))));
  }
  return [xs, ys];
}

function _choice_title(prefix, action) {
  return {
    text: `${prefix}  |  <b>Agent chooses ${action}</b>`,
    font: { size: 15, color: "#1a1d23" },
  };
}

export const STICKINESS_MIN = 0.0;
export const STICKINESS_MAX = 1.0;
export const N_STICKINESS_POINTS = 100;
export const STICKY_BASE_RGB = [31 / 255, 78 / 255, 121 / 255];

export function sticky_delay_rgba(delay_LL, delay_lo, delay_hi) {
  const delay_span = Math.max(delay_hi - delay_lo, 1);
  const t = (delay_LL - delay_lo) / delay_span;
  const lightness = 0.55 - 0.55 * t;
  const color = STICKY_BASE_RGB.map((c) => c + (1.0 - c) * lightness);
  const alpha = 0.7 + 0.3 * t;
  const [r, g, b] = color.map((c) => Math.round(c * 255));
  return `rgba(${r},${g},${b},${alpha.toFixed(3)})`;
}

export function max_ss_resisted(agent, s, delay_SS, delay_LL, reward_LL, horizon, decision_interval) {
  let V_resist_LL = 0.0;
  let V_resist_SS = 0.0;
  let V_dessert_LL = 0.0;
  let V_dessert_SS = 0.0;

  for (let t = 0; t < horizon; t++) {
    const sticking = stick_p_t(s, t);
    const v_SS = agent.value_function(1.0, delay_SS + t * decision_interval);
    const v_LL = agent.value_function(1.0, delay_LL + t * decision_interval);
    V_resist_LL += sticking * v_LL;
    V_resist_SS += (1.0 - sticking) * v_SS;
    V_dessert_SS += sticking * v_SS;
    V_dessert_LL += (1.0 - sticking) * v_LL;
  }

  const denom = V_dessert_SS - V_resist_SS;
  if (denom === 0) return null;
  return (reward_LL * (V_resist_LL - V_dessert_LL)) / denom;
}

export function foreseeable_events(agent, environment, state) {
  return [
    agent.choice_horizon(state, "dessert", environment),
    agent.choice_horizon(state, "resist", environment),
  ];
}

export function sticky_outcome_events(agent, environment, state) {
  const horizon = agent.get_horizon(state, environment);
  const interval = environment.decision_interval;
  const ss_events = [];
  const ll_events = [];
  for (let t = 0; t < horizon; t++) {
    const [, ss_outcomes] = environment.simulate(state, "dessert");
    const [, ll_outcomes] = environment.simulate(state, "resist");
    for (const [reward, delay] of Object.values(ss_outcomes)) {
      ss_events.push([delay + t * interval, reward]);
    }
    for (const [reward, delay] of Object.values(ll_outcomes)) {
      ll_events.push([delay + t * interval, reward]);
    }
  }
  return [ss_events, ll_events];
}

function _value_label(x, y, text, { color, position = "middle left", size = 12, bold = false } = {}) {
  let font = { color, size };
  if (bold) {
    font = {
      color,
      size: Math.max(size + 2, 15),
      family: "Arial Black, Helvetica Neue, sans-serif",
    };
  }
  return {
    type: "scatter",
    mode: "text",
    x: [x],
    y: [y],
    text: [text],
    textposition: position,
    textfont: font,
    showlegend: false,
    hoverinfo: "skip",
  };
}

function _ll_sail_trace(agent, delay, reward, decision_t, { sail_color, label = null, show_legend = true, bold_decision = false } = {}) {
  const label_color = COLORS.resist;
  const [xs, ys] = _dense_sail_xy(agent, reward, delay);
  const tip = agent.value_function(reward, 0);
  const decision_val = agent.value_function(reward, delay - decision_t);
  const decision_label_x = decision_t - LABEL_DISCOUNT_OFFSET;

  const traces = [
    {
      type: "scatter",
      mode: "lines",
      x: xs,
      y: ys,
      line: { color: sail_color, width: 2 },
      name: label || "resist",
      showlegend: show_legend && label != null,
      hovertemplate: "t=%{x}h<br>V=%{y:.3f}<extra></extra>",
    },
    {
      type: "scatter",
      mode: "markers",
      x: [delay],
      y: [tip],
      marker: { size: 9, color: sail_color },
      showlegend: false,
      hoverinfo: "skip",
    },
    _value_label(delay, tip + 0.5, tip.toFixed(2), { color: label_color, position: "top right" }),
    {
      type: "scatter",
      mode: "markers",
      x: [decision_t],
      y: [decision_val],
      marker: {
        size: bold_decision ? 11 : 8,
        color: sail_color,
        line: bold_decision ? { color: sail_color, width: 2 } : { width: 0 },
      },
      showlegend: false,
      hoverinfo: "skip",
    },
    _value_label(decision_label_x, decision_val, decision_val.toFixed(2), {
      color: label_color,
      position: "middle left",
      bold: bold_decision,
    }),
  ];
  const shapes = [
    {
      type: "line",
      x0: delay,
      x1: delay,
      y0: 0,
      y1: 1,
      yref: "paper",
      line: { color: sail_color, width: 1, dash: "dot" },
      opacity: 0.45,
    },
    {
      type: "line",
      x0: decision_t,
      x1: decision_t,
      y0: 0,
      y1: 1,
      yref: "paper",
      line: { color: "black", width: 1, dash: "dot" },
      opacity: 0.55,
    },
  ];
  const y_max = ys.length ? Math.max(...ys) : 0;
  return [traces, shapes, y_max];
}

function _ss_sail_traces(agent, ss_events, { bold_first = false } = {}) {
  const base = COLORS.dessert_sail;
  const label_color = COLORS.dessert;
  const traces = [];
  let y_max = 0;
  const n = ss_events.length;
  for (let i = 0; i < n; i++) {
    const [delay, reward] = ss_events[i];
    const sail_color = fade(base, i, n);
    const emphasize = bold_first && i === 0;
    const tip_label_x = delay - LABEL_DISCOUNT_OFFSET;
    if (delay === 0) {
      const tip = agent.value_function(reward, 0);
      y_max = Math.max(y_max, tip);
      traces.push({
        type: "scatter",
        mode: "markers",
        x: [delay],
        y: [tip],
        marker: {
          size: emphasize ? 13 : 10,
          color: sail_color,
          line: emphasize ? { color: sail_color, width: 2 } : { width: 0 },
        },
        name: "dessert",
        showlegend: i === 0,
        hovertemplate: "SS @ %{x}h<br>V=%{y:.3f}<extra></extra>",
      });
      traces.push(
        _value_label(tip_label_x, tip, tip.toFixed(2), {
          color: label_color,
          position: "middle left",
          bold: emphasize,
        })
      );
      continue;
    }

    const [xs, ys] = _dense_sail_xy(agent, reward, delay);
    const tip = agent.value_function(reward, 0);
    y_max = Math.max(y_max, ys.length ? Math.max(...ys) : 0, tip);
    traces.push({
      type: "scatter",
      mode: "lines",
      x: xs,
      y: ys,
      line: { color: sail_color, width: 2 },
      name: "dessert",
      showlegend: i === 0,
      hovertemplate: "t=%{x}h<br>V=%{y:.3f}<extra></extra>",
    });
    traces.push({
      type: "scatter",
      mode: "markers",
      x: [delay],
      y: [tip],
      marker: {
        size: emphasize ? 13 : 10,
        color: sail_color,
        line: emphasize ? { color: sail_color, width: 2 } : { width: 0 },
      },
      showlegend: false,
      hoverinfo: "skip",
    });
    traces.push(
      _value_label(tip_label_x, tip + 0.5, tip.toFixed(2), {
        color: label_color,
        position: "middle left",
        bold: emphasize,
      })
    );
  }
  return [traces, y_max];
}

function _aggregate_traces(action_values, chosen) {
  const dessert_agg = action_values.dessert;
  const resist_agg = action_values.resist;
  const label_x = -LABEL_AGG_OFFSET;

  function agg_marker(name, y, color) {
    return {
      type: "scatter",
      mode: "markers",
      x: [0],
      y: [y],
      marker: {
        size: 12,
        color,
        opacity: 1.0,
        line: { color, width: 1.2 },
      },
      name: `${name} aggregate`,
      hovertemplate: `${name} agg=%{y:.3f}<extra></extra>`,
    };
  }

  const traces = [
    agg_marker("dessert", dessert_agg, COLORS.dessert),
    _value_label(label_x, dessert_agg + 0.5, dessert_agg.toFixed(2), {
      color: COLORS.dessert,
      position: "middle left",
      size: 13,
      bold: chosen === "dessert",
    }),
    agg_marker("resist", resist_agg, COLORS.resist),
    _value_label(label_x, resist_agg + 0.5, resist_agg.toFixed(2), {
      color: COLORS.resist,
      position: "middle left",
      size: 13,
      bold: chosen === "resist",
    }),
  ];
  return [traces, Math.max(dessert_agg, resist_agg)];
}

function _hour_xticks(x_max) {
  const ticks = [];
  for (let t = 0; t <= x_max; t += 12) ticks.push(t);
  return [ticks, ticks.map(String)];
}

function _assign_axes(traces, { xaxis = "x", yaxis = "y" } = {}) {
  for (const tr of traces) {
    tr.xaxis = xaxis;
    tr.yaxis = yaxis;
  }
  return traces;
}

function _shapes_on_subplot(shapes, { xref = "x2", yref = "y2" } = {}) {
  const out = [];
  for (const shape of shapes) {
    const s = { ...shape };
    s.xref = xref;
    if (s.yref === "paper") s.yref = `${yref} domain`;
    else s.yref = yref;
    out.push(s);
  }
  return out;
}

function _spaced_ys(count, lo, hi) {
  if (count <= 0) return [];
  if (count === 1) return [(lo + hi) / 2];
  const ys = [];
  for (let i = 0; i < count; i++) ys.push(lo + ((hi - lo) * i) / (count - 1));
  return ys;
}

function _hyp_stack_parts(agent, ss_events, ll_events) {
  const dessert_parts = ss_events.map(([delay, reward]) => [delay, agent.value_function(reward, delay)]);
  const resist_parts = ll_events.map(([delay, reward]) => [delay, agent.value_function(reward, delay)]);
  return [dessert_parts, resist_parts];
}

function _sticky_stack_parts(agent, environment, state) {
  const interval = environment.decision_interval;
  const dessert_days = agent.choice_horizon(state, "dessert", environment);
  const resist_days = agent.choice_horizon(state, "resist", environment);
  const dessert_parts = dessert_days.map(([t, ev]) => [t * interval, ev]);
  const resist_parts = resist_days.map(([t, ev]) => [t * interval, ev]);
  return [dessert_parts, resist_parts];
}

function _aggregate_stack_panel(dessert_parts, resist_parts, action_values, chosen) {
  const n = dessert_parts.length;
  if (n === 0) return [[], [], 1.0];

  const dessert_agg = action_values.dessert;
  const resist_agg = action_values.resist;
  const x_dessert = 0.0;
  const x_resist = 1.0;
  const bar_w = 0.55;
  let bottom_s = 0.0;
  let bottom_r = 0.0;
  const heights = [...dessert_parts.map(([, h]) => h), ...resist_parts.map(([, h]) => h)];
  const min_label = heights.length ? 0.05 * Math.max(...heights) : 0.0;

  const traces = [];
  const dessert_callouts = [];
  const resist_callouts = [];

  for (let i = 0; i < n; i++) {
    const [hours_s, h_s] = dessert_parts[i];
    const [hours_r, h_r] = resist_parts[i];
    const edge_s = chosen === "dessert" ? 1.2 : 0.5;
    const edge_r = chosen === "resist" ? 1.2 : 0.5;
    const color_s = fade(COLORS.dessert_sail, i, n);
    const color_r = fade(COLORS.resist_sail, i, n);

    traces.push({
      type: "bar",
      x: [x_dessert],
      y: [h_s],
      base: [bottom_s],
      width: [bar_w],
      marker: { color: color_s, line: { color: "black", width: edge_s } },
      showlegend: false,
      hovertemplate: `dessert @ ${Math.trunc(hours_s)}h<br>V=%{y:.3f}<extra></extra>`,
    });
    traces.push({
      type: "bar",
      x: [x_resist],
      y: [h_r],
      base: [bottom_r],
      width: [bar_w],
      marker: { color: color_r, line: { color: "black", width: edge_r } },
      showlegend: false,
      hovertemplate: `resist @ ${Math.trunc(hours_r)}h<br>V=%{y:.3f}<extra></extra>`,
    });
    if (h_s >= min_label) dessert_callouts.push([hours_s, h_s, bottom_s + h_s / 2]);
    if (h_r >= min_label) resist_callouts.push([hours_r, h_r, bottom_r + h_r / 2]);
    bottom_s += h_s;
    bottom_r += h_r;
  }

  const y_peak = Math.max(dessert_agg, resist_agg) || 1.0;
  const annotations = [];

  function callout_annotations(callouts, x_bar, x_text, { color, xanchor }) {
    const ys = _spaced_ys(callouts.length, y_peak * 0.06, y_peak * 0.94);
    const x_edge = xanchor === "right" ? x_bar - bar_w / 2 : x_bar + bar_w / 2;
    for (let i = 0; i < callouts.length; i++) {
      const [hours, h, mid_y] = callouts[i];
      const y_lab = ys[i];
      annotations.push({
        x: x_edge,
        y: mid_y,
        ax: x_text,
        ay: y_lab,
        xref: "x",
        yref: "y",
        axref: "x",
        ayref: "y",
        text: `${Math.trunc(hours)}h<br>${h.toFixed(2)}`,
        showarrow: true,
        arrowhead: 2,
        arrowsize: 0.9,
        arrowwidth: 1,
        arrowcolor: "#666666",
        font: { size: 10, color },
        xanchor,
        align: xanchor,
        bgcolor: "rgba(255,255,255,0.65)",
        borderpad: 2,
      });
    }
  }

  callout_annotations(dessert_callouts, x_dessert, x_dessert - 0.85, { color: COLORS.dessert, xanchor: "right" });
  callout_annotations(resist_callouts, x_resist, x_resist + 0.85, { color: COLORS.resist, xanchor: "left" });

  annotations.push(
    {
      x: x_dessert - 0.75,
      y: y_peak * 1.04,
      xref: "x",
      yref: "y",
      text: `<b>Σ dessert = ${dessert_agg.toFixed(2)}</b>`,
      showarrow: false,
      font: { size: 12, color: COLORS.dessert },
      xanchor: "left",
      yanchor: "bottom",
    },
    {
      x: x_resist + 0.75,
      y: y_peak * 1.04,
      xref: "x",
      yref: "y",
      text: `<b>Σ resist = ${resist_agg.toFixed(2)}</b>`,
      showarrow: false,
      font: { size: 12, color: COLORS.resist },
      xanchor: "right",
      yanchor: "bottom",
    }
  );

  _assign_axes(traces, { xaxis: "x", yaxis: "y" });
  return [traces, annotations, y_peak];
}

function _sails_dual_panel(
  agent,
  ss_events,
  ll_events,
  action,
  action_values,
  dessert_parts,
  resist_parts,
  { title_prefix, kind = "sails", day = null, extra_meta = null } = {}
) {
  const n = ll_events.length;
  const short_vision = (agent.vision ?? "long") === "short";
  const [stack_traces, stack_annotations, stack_ymax] = _aggregate_stack_panel(
    dessert_parts,
    resist_parts,
    action_values,
    action
  );

  const sail_traces = [];
  let shapes = [];
  let y_max = 0;
  const n_sails = ll_events.length;

  for (let i = 0; i < ll_events.length; i++) {
    const [delay, reward] = ll_events[i];
    const decision_t = ss_events[i][0];
    const [traces_i, shapes_i, peak] = _ll_sail_trace(agent, delay, reward, decision_t, {
      sail_color: fade(COLORS.resist_sail, i, n_sails),
      label: i === 0 ? "resist" : null,
      show_legend: i === 0,
      bold_decision: short_vision && action === "resist" && i === 0,
    });
    sail_traces.push(...traces_i);
    shapes.push(...shapes_i);
    y_max = Math.max(y_max, peak);
  }

  const [ss_traces, ss_ymax] = _ss_sail_traces(agent, ss_events, {
    bold_first: short_vision && action === "dessert",
  });
  sail_traces.push(...ss_traces);
  y_max = Math.max(y_max, ss_ymax);

  if (!short_vision) {
    const [agg_traces, agg_ymax] = _aggregate_traces(action_values, action);
    sail_traces.push(...agg_traces);
    y_max = Math.max(y_max, agg_ymax);
  }

  _assign_axes(sail_traces, { xaxis: "x2", yaxis: "y2" });
  shapes = _shapes_on_subplot(shapes, { xref: "x2", yref: "y2" });

  const x_max = Math.max(
    Math.max(0, ...ll_events.map(([d]) => d)),
    Math.max(0, ...ss_events.map(([d]) => d)),
    1
  );
  const label_pad = short_vision ? LABEL_DISCOUNT_OFFSET + 4 : LABEL_AGG_OFFSET + 6;
  const pad_x = Math.max(label_pad, x_max * 0.12);
  const [xticks, xticklabels] = _hour_xticks(x_max);
  const title = _choice_title(title_prefix, action);

  const stack_domain = [0.0, 0.3];
  const sails_domain = [0.35, 1.0];
  const stack_title_x = 0.5 * (stack_domain[0] + stack_domain[1]);
  const panel_annotations = [
    ...stack_annotations,
    {
      x: stack_title_x,
      y: 1.0,
      xref: "paper",
      yref: "paper",
      text: "Summation of discounted values<br>(Aggregate option values)",
      showarrow: false,
      font: { size: 12, color: "#1a1d23" },
      xanchor: "center",
      yanchor: "top",
      align: "center",
      yshift: -52,
    },
  ];

  const meta = {
    day,
    horizon: n,
    choice: action,
    action_values,
    ll_ss_gap: action_values.resist - action_values.dessert,
  };
  if (extra_meta) Object.assign(meta, extra_meta);

  return {
    kind,
    traces: [...stack_traces, ...sail_traces],
    layout: {
      title,
      barmode: "overlay",
      xaxis: {
        title: "",
        domain: stack_domain,
        range: [-1.25, 2.25],
        tickvals: [0.0, 1.0],
        ticktext: ["dessert", "resist"],
        zeroline: false,
      },
      yaxis: {
        title: "Perceived value of reward",
        domain: [0.0, 0.92],
        // Extra headroom so Σ labels / callouts are not clipped when UI is scaled.
        range: [0, stack_ymax * 1.4],
        gridcolor: "rgba(0,0,0,0.12)",
      },
      xaxis2: {
        title: "Hours from now",
        domain: sails_domain,
        range: [-pad_x, x_max + pad_x],
        tickvals: xticks,
        ticktext: xticklabels,
        anchor: "y2",
      },
      yaxis2: {
        title: "Perceived value",
        domain: [0.0, 0.92],
        range: [0, y_max ? y_max * 1.35 : 1],
        anchor: "x2",
        overlaying: false,
        side: "left",
      },
      annotations: panel_annotations,
      shapes,
      legend: {
        orientation: "h",
        y: -0.16,
        x: sails_domain[0],
        xref: "paper",
      },
      margin: { l: 60, r: 24, t: 80, b: 90 },
      height: 880,
    },
    meta,
  };
}

export function sails_frame(agent, environment, state, { day = null } = {}) {
  const [ss_events, ll_events] = foreseeable_events(agent, environment, state);
  const n = ll_events.length;
  const [action, action_values] = agent.choose_action(state, environment);
  const [dessert_parts, resist_parts] = _hyp_stack_parts(agent, ss_events, ll_events);
  const title_prefix =
    day != null
      ? `Day ${day}  |  Choice sails with a horizon of ${n} days`
      : `Choice sails with a horizon of ${n} days`;
  return _sails_dual_panel(agent, ss_events, ll_events, action, action_values, dessert_parts, resist_parts, {
    title_prefix,
    kind: "sails",
    day,
  });
}

export function sticky_sails_frame(agent, environment, state) {
  const [ss_events, ll_events] = sticky_outcome_events(agent, environment, state);
  const n = ll_events.length;
  const [action, action_values] = agent.choose_action(state, environment);
  const [dessert_parts, resist_parts] = _sticky_stack_parts(agent, environment, state);
  const perceived_s = agent.perceived_s ?? null;
  return _sails_dual_panel(agent, ss_events, ll_events, action, action_values, dessert_parts, resist_parts, {
    title_prefix: `Sticky sails | s=${perceived_s} | Choice sails with a horizon of ${n} days`,
    kind: "sticky_sails",
    extra_meta: { perceived_s },
  });
}


export function sticky_step_frame(agent, environment, state, { day = null, last_info = null } = {}) {
  const beta_belief = environment.beta_belief;
  if (beta_belief == null) throw new Error("sticky_step_frame requires environment.beta_belief");

  const [ss_events, ll_events] = sticky_outcome_events(agent, environment, state);
  const n = agent.get_horizon(state, environment);
  const s_hat = environment.s_hat();
  const perceived_s = agent.perceived_stickiness(environment);
  const prior_mean = beta_belief.prior_mean();
  const learn = Boolean(agent.learn_stickiness);
  const [intent, action_values] = agent.choose_action(state, environment);

  const sail_traces = [];
  let shapes = [];
  let y_max = 0.0;
  const n_sails = ll_events.length;

  for (let i = 0; i < ll_events.length; i++) {
    const [delay, reward] = ll_events[i];
    const decision_t = ss_events[i][0];
    const [traces_i, shapes_i, peak] = _ll_sail_trace(agent, delay, reward, decision_t, {
      sail_color: fade(COLORS.resist_sail, i, n_sails),
      label: i === 0 ? "resist" : null,
      show_legend: i === 0,
    });
    sail_traces.push(...traces_i);
    shapes.push(...shapes_i);
    y_max = Math.max(y_max, peak);
  }

  const [ss_traces, ss_ymax] = _ss_sail_traces(agent, ss_events);
  sail_traces.push(...ss_traces);
  y_max = Math.max(y_max, ss_ymax);

  const [agg_traces, agg_ymax] = _aggregate_traces(action_values, intent);
  sail_traces.push(...agg_traces);
  y_max = Math.max(y_max, agg_ymax);

  _assign_axes(sail_traces, { xaxis: "x", yaxis: "y" });
  shapes = _shapes_on_subplot(shapes, { xref: "x", yref: "y" });

  const interval = environment.decision_interval;
  const horizon_days = agent.horizon != null ? agent.horizon : n;
  const x_max = horizon_days * interval;
  const pad_x = Math.max(8, x_max * 0.12);
  const [xticks, xticklabels] = _hour_xticks(x_max);

  const [xs, dens] = beta_belief.pdf_grid(251);
  const y_top = dens.length ? Math.max(...dens) : 1.0;
  const marker_s = learn ? perceived_s : prior_mean;
  const marker_name = learn ? `ŝ=${marker_s.toFixed(2)}` : `fixed s=α/(α+β)=${prior_mean.toFixed(2)}`;
  const belief_traces = [
    {
      type: "scatter",
      mode: "lines",
      x: xs,
      y: dens,
      line: { color: COLORS.belief, width: 2.0 },
      fill: "tozeroy",
      fillcolor: "rgba(44,160,44,0.18)",
      name: "Beta posterior",
      xaxis: "x2",
      yaxis: "y2",
      hovertemplate: "s=%{x:.3f}<br>density=%{y:.3f}<extra></extra>",
    },
    {
      type: "scatter",
      mode: "lines",
      x: [marker_s, marker_s],
      y: [0.0, y_top ? y_top * 1.15 : 1.0],
      line: { color: COLORS.perceived_s, width: 1.5 },
      name: marker_name,
      xaxis: "x2",
      yaxis: "y2",
      hoverinfo: "skip",
    },
  ];

  const day_bit = day != null ? `Day ${day}  |  ` : "";
  const last_realized = last_info ? last_info.realized : null;
  const last_stuck = last_info ? last_info.stuck : null;
  const last_intent = last_info ? last_info.intent : null;
  let last_aligned = last_info ? last_info.aligned : null;
  if (last_aligned == null && last_realized != null && last_intent != null) {
    last_aligned = last_realized === last_intent;
  }

  let last_action_html;
  let last_action_label;
  if (last_realized == null) {
    last_action_html = "Last action: —";
    last_action_label = null;
  } else {
    let roll;
    if (last_stuck == null) roll = "free";
    else if (last_stuck) roll = "stick";
    else roll = "switch";
    let align_html;
    let align_label;
    if (last_aligned) {
      align_html = '<span style="color:#2e7d32;font-weight:600;">aligned with intent</span>';
      align_label = "aligned";
    } else {
      align_html = `<span style="color:#c62828;font-weight:600;">diverged from intent (${last_intent})</span>`;
      align_label = `diverged (wanted ${last_intent})`;
    }
    last_action_html = `Last action: <b>${last_realized}</b> (${roll}) · ${align_html}`;
    last_action_label = `${last_realized} (${roll}, ${align_label})`;
  }

  let s_line;
  if (learn) s_line = `Perceived stickiness=${perceived_s.toFixed(3)} (learning)`;
  else {
    s_line =
      `Fixed stickiness=${prior_mean.toFixed(3)} ` +
      `(prior mean α/(α+β), α=${Number(beta_belief.alpha0)} β=${Number(beta_belief.beta0)})`;
  }

  const title = {
    text: `${day_bit}${s_line}<br>horizon=${n}  |  <b>Agent would choose ${intent}</b><br>${last_action_html}`,
    font: { size: 15, color: "#1a1d23" },
  };

  const sails_domain = [0.0, 0.66];
  const belief_domain = [0.72, 1.0];
  const belief_title =
    `Beta Distribution<br>` +
    `α=${beta_belief.alpha.toFixed(1)} | β=${beta_belief.beta.toFixed(1)}<br>` +
    `sticks=${beta_belief.n_stick()} | switches=${beta_belief.n_switch()}`;

  const meta = {
    day,
    horizon: n,
    choice: intent,
    action_values,
    ll_ss_gap: action_values.resist - action_values.dessert,
    perceived_s,
    s_hat,
    prior_mean,
    prior_alpha: beta_belief.alpha0,
    prior_beta: beta_belief.beta0,
    learn_stickiness: learn,
    alpha: beta_belief.alpha,
    beta: beta_belief.beta,
    n_stick: beta_belief.n_stick(),
    n_switch: beta_belief.n_switch(),
    last_action_label,
  };
  if (last_info) {
    meta.last_intent = last_intent;
    meta.last_realized = last_realized;
    meta.last_stuck = last_stuck;
    meta.last_aligned = last_aligned;
  }

  return {
    kind: "sticky_step",
    traces: [...sail_traces, ...belief_traces],
    layout: {
      title,
      xaxis: {
        title: "Hours from now",
        domain: sails_domain,
        range: [-pad_x, x_max + pad_x],
        tickvals: xticks,
        ticktext: xticklabels,
        anchor: "y",
      },
      yaxis: {
        title: "Perceived value",
        domain: [0.0, 0.88],
        range: [0, y_max ? y_max * 1.25 : 1],
        anchor: "x",
      },
      xaxis2: {
        title: "Stickiness s",
        domain: belief_domain,
        range: [0.0, 1.0],
        anchor: "y2",
      },
      yaxis2: {
        title: "Density",
        domain: [0.0, 0.88],
        range: [0.0, y_top ? y_top * 1.15 : 1.0],
        anchor: "x2",
      },
      annotations: [
        {
          x: 0.5 * (belief_domain[0] + belief_domain[1]),
          y: 1.0,
          xref: "paper",
          yref: "paper",
          text: belief_title,
          showarrow: false,
          font: { size: 12, color: "#1a1d23" },
          xanchor: "center",
          yanchor: "top",
          align: "center",
          yshift: -8,
        },
      ],
      shapes,
      legend: { orientation: "h", y: -0.16, x: 0.0, xref: "paper" },
      margin: { l: 60, r: 24, t: 92, b: 90 },
      height: 700,
    },
    meta,
  };
}

export function aggregate_frame(agent, environment, state) {
  const [action, action_values] = agent.choose_action(state, environment);
  const ss_events = agent.choice_horizon(state, "dessert", environment);
  const ll_events = agent.choice_horizon(state, "resist", environment);
  const ll_delay = Number(environment.rewards.LL.delay);

  function bundle_value_at(events, at_time) {
    let total = 0;
    for (const [delay, reward] of events) {
      if (delay >= at_time) total += agent.value_function(reward, delay - at_time);
    }
    return total;
  }

  let xs, ys;
  if (ll_delay <= 0) {
    xs = [0.0];
    ys = [Number(bundle_value_at(ll_events, 0))];
  } else {
    const n = SAIL_SAMPLE_POINTS;
    xs = [];
    ys = [];
    for (let i = 0; i < n; i++) {
      const t = (ll_delay * i) / (n - 1);
      xs.push(t);
      ys.push(Number(bundle_value_at(ll_events, t)));
    }
  }
  const ss_value = bundle_value_at(ss_events, 0);
  const perceived_ll = bundle_value_at(ll_events, 0);
  const ll_tip = bundle_value_at(ll_events, ll_delay);
  const y_max = Math.max(ys.length ? Math.max(...ys) : 0, ss_value, perceived_ll, ll_tip);

  const resist_chosen = action === "resist";
  const dessert_chosen = action === "dessert";
  const resist_w = resist_chosen ? 3.5 : 2.0;
  const dessert_size = dessert_chosen ? 16 : 10;
  const resist_size = resist_chosen ? 14 : 10;

  const traces = [
    {
      type: "scatter",
      mode: "lines",
      x: xs,
      y: ys,
      line: { color: COLORS.resist, width: resist_w },
      opacity: 1.0,
      name: "resist" + (resist_chosen ? " ★" : ""),
      hovertemplate: "t=%{x}h<br>V=%{y:.3f}<extra></extra>",
    },
    {
      type: "scatter",
      mode: "markers",
      x: [0],
      y: [perceived_ll],
      marker: {
        size: resist_size,
        color: COLORS.resist,
        opacity: 1.0,
        line: { color: COLORS.resist, width: resist_chosen ? 2 : 0 },
      },
      showlegend: false,
      hoverinfo: "skip",
    },
    _value_label(0, perceived_ll, perceived_ll.toFixed(2), { color: COLORS.resist, position: "top left" }),
    {
      type: "scatter",
      mode: "markers",
      x: [ll_delay],
      y: [ll_tip],
      marker: { size: resist_size, color: COLORS.resist, opacity: 1.0 },
      showlegend: false,
      hoverinfo: "skip",
    },
    _value_label(ll_delay, ll_tip, ll_tip.toFixed(2), { color: COLORS.resist, position: "top right" }),
    {
      type: "scatter",
      mode: "markers",
      x: [0],
      y: [ss_value],
      marker: {
        size: dessert_size,
        color: COLORS.dessert,
        opacity: 1.0,
        line: { color: COLORS.dessert, width: dessert_chosen ? 2.5 : 0 },
      },
      name: "dessert" + (dessert_chosen ? " ★" : ""),
      hovertemplate: "dessert=%{y:.3f}<extra></extra>",
    },
    _value_label(0, ss_value, ss_value.toFixed(2), { color: COLORS.dessert, position: "bottom left" }),
  ];

  if (resist_chosen) {
    traces[2].textfont = {
      color: COLORS.resist,
      size: 14,
      family: "Arial Black, Helvetica Neue, sans-serif",
    };
  }
  if (dessert_chosen) {
    traces[traces.length - 1].textfont = {
      color: COLORS.dessert,
      size: 14,
      family: "Arial Black, Helvetica Neue, sans-serif",
    };
  }

  const shapes = [
    {
      type: "line",
      x0: ll_delay,
      x1: ll_delay,
      y0: 0,
      y1: 1,
      yref: "paper",
      line: { color: COLORS.resist, width: 1, dash: "dot" },
      opacity: 0.5,
    },
  ];
  const pad = Math.max(2, ll_delay * 0.2);

  return {
    kind: "aggregate_sail",
    traces,
    layout: {
      title: _choice_title("Bundled value on day 1", action),
      xaxis: { title: "Hours from now", range: [-pad, ll_delay + Math.max(1, ll_delay * 0.15)] },
      yaxis: { title: "Perceived value", range: [0, y_max ? y_max * 1.15 : 1] },
      shapes,
      legend: { orientation: "h", y: -0.15 },
      margin: { l: 60, r: 20, t: 50, b: 70 },
      height: 620,
    },
    meta: { choice: action, action_values },
  };
}

export function hyperbolic_frame(reward, initial_delay, discount_factor, { elapsed = 0 } = {}) {
  initial_delay = Math.trunc(initial_delay);
  elapsed = Math.max(0, Math.min(Math.trunc(elapsed), initial_delay));
  const remaining = initial_delay - elapsed;
  const k = Number(discount_factor);
  const V = (distance) => reward / (1.0 + k * distance);
  const [xs, ys] = _dense_hyperbolic_xy(reward, remaining, k);
  const perceived = V(remaining);
  const x_hi = Math.max(initial_delay, 1);
  // Extra left pad so the day-0 value label is not clipped.
  const x_lo = -Math.max(2.0, x_hi * 0.12);
  const y_hi = Number(reward) + 1.0;

  return {
    kind: "hyperbolic_curve",
    traces: [
      {
        type: "scatter",
        mode: "lines",
        x: xs,
        y: ys,
        line: { color: COLORS.resist, width: 2.5 },
        name: "hyperbolic sail",
        hovertemplate: "t=%{x}<br>V=%{y:.3f}<extra></extra>",
      },
      {
        type: "scatter",
        mode: "markers+text",
        x: [0],
        y: [perceived],
        text: [perceived.toFixed(2)],
        textposition: "top center",
        textfont: { color: "#1a1d23", size: 13 },
        marker: { size: 12, color: "black", opacity: 1.0 },
        name: "perceived at day 0",
        hovertemplate: "perceived=%{y:.3f}<extra></extra>",
      },
    ],
    layout: {
      title: {
        text: `Reward approaching | Time elapsed = ${elapsed}h  |  remaining = ${remaining}h  |  <b>V(0) = ${perceived.toFixed(2)}</b>`,
        font: { size: 15, color: "#1a1d23" },
      },
      xaxis: {
        title: "Hours from approaching reward",
        range: [x_lo, x_hi],
        autorange: false,
        fixedrange: true,
      },
      yaxis: {
        title: "Perceived value",
        range: [0, y_hi],
        autorange: false,
        fixedrange: true,
      },
      shapes: [
        {
          type: "line",
          x0: remaining,
          x1: remaining,
          y0: 0,
          y1: 1,
          yref: "paper",
          line: { color: "#888", width: 1, dash: "dash" },
        },
      ],
      legend: { orientation: "h", y: -0.15 },
      margin: { l: 72, r: 20, t: 50, b: 70 },
      height: 620,
      uirevision: `hyp|${reward}|${initial_delay}|${k}`,
    },
    meta: {
      reward,
      initial_delay,
      elapsed,
      remaining,
      discount_factor: k,
      perceived,
      finished: elapsed >= initial_delay,
    },
  };
}

export function ss_delay_from_ll(ll_delay) {
  return Math.max(0, Math.round(0.75 * Math.trunc(ll_delay)));
}

export function hyperbolic_ss_ll_frame(
  ss_value,
  ll_value,
  ll_delay,
  discount_factor,
  { elapsed = 0, ss_delay = null } = {}
) {
  ss_value = Number(ss_value);
  ll_value = Number(ll_value);
  ll_delay = Math.trunc(ll_delay);
  const ss_delay_explicit = ss_delay != null && ss_delay !== "";
  ss_delay = ss_delay_explicit
    ? Math.max(0, Math.trunc(Number(ss_delay)))
    : ss_delay_from_ll(ll_delay);
  const k = Number(discount_factor);
  const max_delay = Math.max(ss_delay, ll_delay);
  elapsed = Math.max(0, Math.min(Math.trunc(elapsed), max_delay));

  const V = (reward, distance) => reward / (1.0 + k * distance);
  const rem_ss = Math.max(0, ss_delay - elapsed);
  const rem_ll = Math.max(0, ll_delay - elapsed);
  const ss_available = elapsed <= ss_delay;
  const ll_available = elapsed <= ll_delay;
  const v_ss = ss_available ? V(ss_value, rem_ss) : null;
  const v_ll = ll_available ? V(ll_value, rem_ll) : null;

  let preference;
  if (ss_available && ll_available) {
    if (Math.abs(v_ss - v_ll) < 1e-12) preference = "indifferent";
    else if (v_ss > v_ll) preference = "SS";
    else preference = "LL";
  } else if (ss_available) preference = "SS";
  else if (ll_available) preference = "LL";
  else preference = "none";

  const [xs_ss, ys_ss] = ss_available ? _dense_hyperbolic_xy(ss_value, rem_ss, k) : [[0.0], [ss_value]];
  const [xs_ll, ys_ll] = ll_available ? _dense_hyperbolic_xy(ll_value, rem_ll, k) : [[0.0], [ll_value]];
  const x_hi = Math.max(max_delay, 1);
  const y_hi = Math.max(ss_value, ll_value) + 1.0;

  function option_style(name, available) {
    if (!available) {
      return { width: 1.2, marker: 7, opacity: 0.2, dash: "dot", suffix: " (delivered)", chosen: false };
    }
    const chosen = preference === "indifferent" || preference === name;
    return {
      width: chosen ? 3.6 : 1.8,
      marker: chosen ? 14 : 9,
      opacity: chosen ? 1.0 : 0.45,
      dash: "solid",
      suffix: preference === "indifferent" ? " =" : chosen ? " ★" : "",
      chosen,
    };
  }

  const ss_style = option_style("SS", ss_available);
  const ll_style = option_style("LL", ll_available);

  function now_label(name, value, available) {
    if (!available) return `${name} delivered`;
    return `${name} ${value.toFixed(2)}`;
  }

  const traces = [
    {
      type: "scatter",
      mode: "lines",
      x: xs_ss,
      y: ys_ss,
      line: { color: COLORS.dessert, width: ss_style.width, dash: ss_style.dash },
      opacity: ss_style.opacity,
      name: `SS sail${ss_style.suffix}`,
      hovertemplate: "SS t=%{x}<br>V=%{y:.3f}<extra></extra>",
    },
    {
      type: "scatter",
      mode: "markers+text",
      x: [0],
      y: [ss_available ? v_ss : ss_value],
      text: [now_label("SS", ss_available ? v_ss : ss_value, ss_available)],
      textposition: "top center",
      textfont: {
        color: COLORS.dessert,
        size: preference === "SS" ? 14 : 12,
        family: preference === "SS" ? "Arial Black, sans-serif" : "Arial, sans-serif",
      },
      marker: {
        size: ss_style.marker,
        color: COLORS.dessert,
        opacity: ss_style.opacity,
        line: { width: preference === "SS" ? 2 : 0, color: "#1a1d23" },
      },
      name: `SS at now${ss_style.suffix}`,
      hovertemplate: ss_available ? `SS V(0)=${v_ss.toFixed(3)}<extra></extra>` : "SS delivered<extra></extra>",
      showlegend: false,
    },
    {
      type: "scatter",
      mode: "lines",
      x: xs_ll,
      y: ys_ll,
      line: { color: COLORS.resist, width: ll_style.width, dash: ll_style.dash },
      opacity: ll_style.opacity,
      name: `LL sail${ll_style.suffix}`,
      hovertemplate: "LL t=%{x}<br>V=%{y:.3f}<extra></extra>",
    },
    {
      type: "scatter",
      mode: "markers+text",
      x: [0],
      y: [ll_available ? v_ll : ll_value],
      text: [now_label("LL", ll_available ? v_ll : ll_value, ll_available)],
      textposition:
        ss_available && ll_available && Math.abs(v_ss - v_ll) < 0.35 ? "bottom center" : "top center",
      textfont: {
        color: COLORS.resist,
        size: preference === "LL" ? 14 : 12,
        family: preference === "LL" ? "Arial Black, sans-serif" : "Arial, sans-serif",
      },
      marker: {
        size: ll_style.marker,
        color: COLORS.resist,
        opacity: ll_style.opacity,
        line: { width: preference === "LL" ? 2 : 0, color: "#1a1d23" },
      },
      name: `LL at now${ll_style.suffix}`,
      hovertemplate: ll_available ? `LL V(0)=${v_ll.toFixed(3)}<extra></extra>` : "LL delivered<extra></extra>",
      showlegend: false,
    },
  ];

  const shapes = [];
  if (ss_available) {
    shapes.push({
      type: "line",
      x0: rem_ss,
      x1: rem_ss,
      y0: 0,
      y1: 1,
      yref: "paper",
      line: { color: COLORS.dessert, width: 1, dash: "dash" },
      opacity: 0.55,
    });
  }
  if (ll_available) {
    shapes.push({
      type: "line",
      x0: rem_ll,
      x1: rem_ll,
      y0: 0,
      y1: 1,
      yref: "paper",
      line: { color: COLORS.resist, width: 1, dash: "dash" },
      opacity: 0.55,
    });
  }

  let pref_html;
  if (preference === "indifferent") pref_html = "<b>indifferent</b>";
  else if (preference === "none") pref_html = "<b>both delivered</b>";
  else pref_html = `<b>prefers ${preference}</b>`;

  return {
    kind: "hyperbolic_ss_ll",
    traces,
    layout: {
      title: {
        text: `Rewards approaching | Time elapsed = ${elapsed}h  |  SS rem = ${rem_ss}h  ·  LL rem = ${rem_ll}h  |  ${pref_html}`,
        font: { size: 15, color: "#1a1d23" },
      },
      xaxis: {
        title: "Hours from approaching reward",
        range: [-0.2, x_hi],
        autorange: false,
        fixedrange: true,
      },
      yaxis: {
        title: "Perceived value",
        range: [0, y_hi],
        autorange: false,
        fixedrange: true,
      },
      shapes,
      legend: { orientation: "h", y: -0.15 },
      margin: { l: 60, r: 20, t: 70, b: 70 },
      height: 620,
      uirevision: `hyp_ss_ll|${ss_value}|${ss_delay}|${ll_value}|${ll_delay}|${k}`,
    },
    meta: {
      ss_value,
      ss_delay,
      ll_value,
      ll_delay,
      discount_factor: k,
      elapsed,
      remaining_ss: rem_ss,
      remaining_ll: rem_ll,
      ss_available,
      ll_available,
      v_ss,
      v_ll,
      preference,
      finished: elapsed >= max_delay,
      ss_delay_rule: ss_delay_explicit
        ? "explicit"
        : "0.75 * ll_delay (rounded)",
    },
  };
}

export function stickiness_axis() {
  const step = (STICKINESS_MAX - STICKINESS_MIN) / (N_STICKINESS_POINTS - 1);
  const xs = [];
  for (let i = 0; i < N_STICKINESS_POINTS; i++) xs.push(STICKINESS_MIN + i * step);
  return xs;
}

export function delay_list(delay_max, delay_step) {
  delay_max = Math.max(0, Math.trunc(delay_max));
  delay_step = Math.max(1, Math.trunc(delay_step));
  const out = [];
  for (let d = 0; d <= delay_max; d += delay_step) out.push(d);
  return out;
}

export function sticky_curves(agent, environment, state, delay_max, delay_step, reward_LL = null) {
  const horizon = agent.get_horizon(state, environment);
  const decision_interval = environment.decision_interval;
  const delay_SS = environment.rewards.SS.delay;
  const current_delay_LL = Math.trunc(environment.rewards.LL.delay);
  if (reward_LL == null) reward_LL = environment.rewards.LL.value;

  let delays = delay_list(delay_max, delay_step);
  if (!delays.includes(current_delay_LL)) {
    delays = [...delays, current_delay_LL].sort((a, b) => a - b);
  }
  const xs = stickiness_axis();
  const curves = [];
  for (const d of delays) {
    const max_ss = [];
    for (const s of xs) {
      let val;
      try {
        val = max_ss_resisted(agent, s, delay_SS, d, reward_LL, horizon, decision_interval);
      } catch (e) {
        val = null;
      }
      max_ss.push(val != null && Math.abs(val) !== Infinity ? val : null);
    }
    curves.push({ delay_LL: d, max_ss });
  }

  let y_peak = 0;
  for (const c of curves) {
    for (const v of c.max_ss) {
      if (v != null && v > y_peak) y_peak = v;
    }
  }
  const ymax = y_peak ? Math.max(8, y_peak * 1.15) : 8;

  const focus_delays = new Set([current_delay_LL]);
  const idx = delays.indexOf(current_delay_LL);
  if (idx > 0) focus_delays.add(delays[idx - 1]);
  if (idx >= 0 && idx < delays.length - 1) focus_delays.add(delays[idx + 1]);
  const focus_vals = [];
  for (const c of curves) {
    if (!focus_delays.has(c.delay_LL)) continue;
    for (const v of c.max_ss) if (v != null) focus_vals.push(v);
  }

  let y_lo, y_hi;
  if (focus_vals.length) {
    const lo = Math.min(...focus_vals);
    const hi = Math.max(...focus_vals);
    const pad = Math.max((hi - lo) * 0.45, 1.5);
    y_lo = Math.max(0.0, lo - pad);
    y_hi = hi + pad;
    const min_span = Math.max(ymax * 0.55, 4.0);
    if (y_hi - y_lo < min_span) {
      const mid = 0.5 * (y_lo + y_hi);
      y_lo = Math.max(0.0, mid - 0.5 * min_span);
      y_hi = y_lo + min_span;
    }
    if (y_hi <= y_lo) y_hi = y_lo + 1.0;
  } else {
    y_lo = 0.0;
    y_hi = ymax;
  }

  const delay_lo = delays.length ? Math.min(...delays) : 0;
  const delay_hi = delays.length ? Math.max(...delays) : 1;
  const traces = [];
  let current_trace_index = null;
  for (let i = 0; i < curves.length; i++) {
    const c = curves[i];
    const is_current = c.delay_LL === current_delay_LL;
    if (is_current) current_trace_index = i;
    traces.push({
      type: "scatter",
      mode: "lines",
      x: xs,
      y: c.max_ss,
      line: {
        width: is_current ? 3.2 : 2.5,
        color: is_current ? "#000000" : sticky_delay_rgba(c.delay_LL, delay_lo, delay_hi),
      },
      name: is_current ? `LL delay = ${c.delay_LL}h (current)` : `LL delay = ${c.delay_LL}h`,
      hovertemplate:
        "s=%{x:.2f}<br>max SS=%{y:.3f}" +
        (is_current ? "<br><i>Click → set sticky sails s</i>" : "") +
        "<extra></extra>",
      // customdata survives Plotly click events more reliably than meta alone
      customdata: xs.map(() => (is_current ? "current" : "other")),
      meta: { is_current },
    });
  }

  return {
    kind: "sticky",
    traces,
    layout: {
      title: `Max SS resisted | LL=${reward_LL}, current delay=${current_delay_LL}h, horizon=${horizon}, discount_factor=${agent.discount_factor}`,
      xaxis: { title: "Perceived Stickiness (s)", range: [0, 1], autorange: false, fixedrange: true },
      yaxis: { title: "Max SS resisted", range: [y_lo, y_hi], autorange: false },
      legend: { orientation: "h", y: -0.22 },
      margin: { l: 60, r: 20, t: 50, b: 80 },
      height: 520,
      hovermode: "closest",
    },
    meta: {
      stickiness: xs,
      delays,
      current_delay_LL,
      current_trace_index,
      reward_LL,
      horizon,
      discount_factor: agent.discount_factor,
      y_range: [y_lo, y_hi],
    },
  };
}
