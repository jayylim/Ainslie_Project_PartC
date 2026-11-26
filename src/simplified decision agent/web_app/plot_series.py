"""
JSON series builders for the unified demo web app.

Pure helpers: turn agent/env state into Plotly-ready payloads. No matplotlib.
Math matches the corresponding visual_demos/*.py scripts.

Sticky solve mirrors sticky_graph.max_ss_resisted (duplicated here so importing
this module does not run sticky_graph's module-level matplotlib script).
"""

from learning.bayesian_stickiness import BayesianStickinessBelief
from sticky_decisions import stick_p_t

# ---------------------------------------------------------------------------
# Shared styling across demos
# Colours, horizon fade, label offsets, and plot titles used across frames.
# ---------------------------------------------------------------------------

COLORS = {
    "dessert": "#c44e52",
    "resist": "#4c72b0",
    # Sail bases; later foresight days fade toward white (labels stay opaque)
    "dessert_sail": "#c44e52",
    "resist_sail": "#4c72b0",
    # sticky_step Beta belief panel
    "belief": "#2ca02c",
    "true_s": "#111111",
    "perceived_s": "#ff7f0e",
}


def fade(base_hex, i, n, max_fade=0.72):
    """i=0 full base colour; i=n-1 faded toward white. Scales with horizon n."""
    h = base_hex.lstrip("#")
    r, g, b = (int(h[j : j + 2], 16) for j in (0, 2, 4))
    t = (i / (n - 1) if n > 1 else 0.0) * max_fade
    r = int(round(r + (255 - r) * t))
    g = int(round(g + (255 - g) * t))
    b = int(round(b + (255 - b) * t))
    return f"#{r:02x}{g:02x}{b:02x}"

# Hours left of the marker for left-side numerals.
# Aggregates sit further left than per-outcome discounted values.
LABEL_DISCOUNT_OFFSET = 2
LABEL_AGG_OFFSET = 5
# Dense samples for display-only hyperbolic sails (tips / maths stay exact).
SAIL_SAMPLE_POINTS = 200


def _dense_sail_xy(agent, reward, delay, *, n_points=SAIL_SAMPLE_POINTS):
    """Smooth sail from t=0 to delivery using agent.value_function."""
    delay = float(delay)
    if delay <= 0:
        return [0.0], [float(agent.value_function(reward, 0))]
    n = max(int(n_points), 2)
    xs = [delay * i / (n - 1) for i in range(n)]
    ys = [float(agent.value_function(reward, delay - t)) for t in xs]
    return xs, ys


def _dense_hyperbolic_xy(reward, remaining, k, *, n_points=SAIL_SAMPLE_POINTS):
    """Smooth sail V = reward / (1 + k * distance); no agent object needed."""
    remaining = float(remaining)
    if remaining <= 0:
        return [0.0], [float(reward)]
    n = max(int(n_points), 2)
    xs = [remaining * i / (n - 1) for i in range(n)]
    ys = [float(reward / (1.0 + k * (remaining - t))) for t in xs]
    return xs, ys


def _choice_title(prefix, action):
    """Plotly HTML title with fully opaque bold choice callout."""
    return {
        "text": f"{prefix}  |  <b>Agent chooses {action}</b>",
        "font": {"size": 15, "color": "#1a1d23"},
    }

# ---------------------------------------------------------------------------
# Sticky-threshold maths (sticky_graph)
# Critical SS solve and delay colour ramp for the max-SS-resisted family.
# ---------------------------------------------------------------------------

STICKINESS_MIN, STICKINESS_MAX, N_STICKINESS_POINTS = 0.0, 1.0, 100

STICKY_BASE_RGB = (31 / 255, 78 / 255, 121 / 255)  # #1f4e79


def sticky_delay_rgba(delay_LL, delay_lo, delay_hi):
    """Same lightness/alpha ramp as sticky_graph.py → Plotly rgba string."""
    delay_span = max(delay_hi - delay_lo, 1)
    t = (delay_LL - delay_lo) / delay_span  # 0 = shortest, 1 = longest
    lightness = 0.55 - 0.55 * t
    color = tuple(c + (1.0 - c) * lightness for c in STICKY_BASE_RGB)
    alpha = 0.70 + 0.30 * t
    r, g, b = (int(round(c * 255)) for c in color)
    return f"rgba({r},{g},{b},{alpha:.3f})"


def max_ss_resisted(agent, s, delay_SS, delay_LL, reward_LL, horizon, decision_interval):
    """Identical derivation to sticky_graph.max_ss_resisted."""
    V_resist_LL = 0.0
    V_resist_SS = 0.0
    V_dessert_LL = 0.0
    V_dessert_SS = 0.0

    for t in range(horizon):
        sticking = stick_p_t(s, t)
        v_SS = agent.value_function(1.0, delay_SS + t * decision_interval)
        v_LL = agent.value_function(1.0, delay_LL + t * decision_interval)

        V_resist_LL += sticking * v_LL
        V_resist_SS += (1.0 - sticking) * v_SS
        V_dessert_SS += sticking * v_SS
        V_dessert_LL += (1.0 - sticking) * v_LL

    denom = V_dessert_SS - V_resist_SS
    if denom == 0:
        return None
    return reward_LL * (V_resist_LL - V_dessert_LL) / denom


# ---------------------------------------------------------------------------
# Build (delay, reward) outcomes for each agent/environment type within horizon (including constructing raw outcomes from sticky EVs)
# ---------------------------------------------------------------------------

def foreseeable_events(agent, environment, state):
    return (
        agent.choice_horizon(state, "dessert", environment),
        agent.choice_horizon(state, "resist", environment),
    )


def sticky_outcome_events(agent, environment, state):
    """(delay, reward) SS / LL pairs at each foresight day.

    Same delay structure as StickyHypAgent.day_value / visualise_sticky_sails:
    outcomes are shifted by t * decision_interval from the current decision.
    (StickyHypAgent.choice_horizon returns sticky EVs, not raw outcomes.)
    """
    horizon = agent.get_horizon(state, environment)
    interval = environment.decision_interval
    ss_events = []
    ll_events = []
    for t in range(horizon):
        _, ss_outcomes = environment.simulate(state, "dessert")
        _, ll_outcomes = environment.simulate(state, "resist")
        for reward, delay in ss_outcomes.values():
            ss_events.append((delay + t * interval, reward))
        for reward, delay in ll_outcomes.values():
            ll_events.append((delay + t * interval, reward))
    return ss_events, ll_events


# ---------------------------------------------------------------------------
# Sail & aggregate-bar-graph drawing helpers (markers, value labels, shapes)
# ---------------------------------------------------------------------------

def _value_label(x, y, text, *, color, position="middle left", size=12, bold=False):
    """Numeral only (no marker). Labels stay fully opaque; sails may be faded."""
    font = {"color": color, "size": size}
    if bold:
        font = {
            "color": color,
            "size": max(size + 2, 15),
            "family": "Arial Black, Helvetica Neue, sans-serif",
        }
    return {
        "type": "scatter",
        "mode": "text",
        "x": [x],
        "y": [y],
        "text": [text],
        "textposition": position,
        "textfont": font,
        "showlegend": False,
        "hoverinfo": "skip",
    }

# visual characteristics of LL sails and tips
def _ll_sail_trace(
    agent,
    delay,
    reward,
    decision_t,
    *,
    sail_color,
    label=None,
    show_legend=True,
    bold_decision=False,
):
    """One LL hyperbolic sail + tip / decision-node markers.

    Sail/marker colour may be horizon-faded; numerals stay opaque resist.
    ``bold_decision`` emphasises the value at the current choice node (used for
    short-vision preference when aggregates are hidden).
    """
    label_color = COLORS["resist"]
    xs, ys = _dense_sail_xy(agent, reward, delay)
    tip = agent.value_function(reward, 0)
    decision_val = agent.value_function(reward, delay - decision_t)
    # Discounted value at decision node — just left of marker
    decision_label_x = decision_t - LABEL_DISCOUNT_OFFSET

    traces = [
        {
            "type": "scatter",
            "mode": "lines",
            "x": xs,
            "y": ys,
            "line": {"color": sail_color, "width": 2},
            "name": label or "resist",
            "showlegend": show_legend and label is not None,
            "hovertemplate": "t=%{x}h<br>V=%{y:.3f}<extra></extra>",
        },
        {
            "type": "scatter",
            "mode": "markers",
            "x": [delay],
            "y": [tip],
            "marker": {"size": 9, "color": sail_color},
            "showlegend": False,
            "hoverinfo": "skip",
        },
        _value_label(
            delay, tip + 0.5, f"{tip:.2f}", color=label_color, position="top right"
        ),
        {
            "type": "scatter",
            "mode": "markers",
            "x": [decision_t],
            "y": [decision_val],
            "marker": {
                "size": 11 if bold_decision else 8,
                "color": sail_color,
                "line": (
                    {"color": sail_color, "width": 2}
                    if bold_decision
                    else {"width": 0}
                ),
            },
            "showlegend": False,
            "hoverinfo": "skip",
        },
        _value_label(
            decision_label_x,
            decision_val,
            f"{decision_val:.2f}",
            color=label_color,
            position="middle left",
            bold=bold_decision,
        ),
    ]
    shapes = [
        {
            "type": "line",
            "x0": delay,
            "x1": delay,
            "y0": 0,
            "y1": 1,
            "yref": "paper",
            "line": {"color": sail_color, "width": 1, "dash": "dot"},
            "opacity": 0.45,
        },
        {
            "type": "line",
            "x0": decision_t,
            "x1": decision_t,
            "y0": 0,
            "y1": 1,
            "yref": "paper",
            "line": {"color": "black", "width": 1, "dash": "dot"},
            "opacity": 0.55,
        },
    ]
    y_max = max(ys) if ys else 0
    return traces, shapes, y_max

# visual characteristics of SS sails and tips
def _ss_sail_traces(agent, ss_events, *, bold_first=False):
    """SS at the current decision is a tip scatter; later SS get sails back to t=0.

    Sail colour fades with foresight index; numerals stay opaque dessert.
    ``bold_first`` emphasises the current-decision SS tip (short-vision preference
    when aggregates are hidden).
    """
    base = COLORS["dessert_sail"]
    label_color = COLORS["dessert"]
    traces = []
    y_max = 0
    n = len(ss_events)
    for i, (delay, reward) in enumerate(ss_events):
        sail_color = fade(base, i, n)
        emphasize = bold_first and i == 0
        # Labels sit left of the tip so they stay clear of crossing sails
        tip_label_x = delay - LABEL_DISCOUNT_OFFSET
        if delay == 0:
            tip = agent.value_function(reward, 0)
            y_max = max(y_max, tip)
            traces.append(
                {
                    "type": "scatter",
                    "mode": "markers",
                    "x": [delay],
                    "y": [tip],
                    "marker": {
                        "size": 13 if emphasize else 10,
                        "color": sail_color,
                        "line": (
                            {"color": sail_color, "width": 2}
                            if emphasize
                            else {"width": 0}
                        ),
                    },
                    "name": "dessert",
                    "showlegend": i == 0,
                    "hovertemplate": "SS @ %{x}h<br>V=%{y:.3f}<extra></extra>",
                }
            )
            traces.append(
                _value_label(
                    tip_label_x,
                    tip,
                    f"{tip:.2f}",
                    color=label_color,
                    position="middle left",
                    bold=emphasize,
                )
            )
            continue

        xs, ys = _dense_sail_xy(agent, reward, delay)
        tip = agent.value_function(reward, 0)
        y_max = max(y_max, max(ys) if ys else 0, tip)
        traces.append(
            {
                "type": "scatter",
                "mode": "lines",
                "x": xs,
                "y": ys,
                "line": {"color": sail_color, "width": 2},
                "name": "dessert",
                "showlegend": i == 0,
                "hovertemplate": "t=%{x}h<br>V=%{y:.3f}<extra></extra>",
            }
        )
        traces.append(
            {
                "type": "scatter",
                "mode": "markers",
                "x": [delay],
                "y": [tip],
                "marker": {
                    "size": 13 if emphasize else 10,
                    "color": sail_color,
                    "line": (
                        {"color": sail_color, "width": 2}
                        if emphasize
                        else {"width": 0}
                    ),
                },
                "showlegend": False,
                "hoverinfo": "skip",
            }
        )
        traces.append(
            _value_label(
                tip_label_x,
                tip + 0.5,
                f"{tip:.2f}",
                color=label_color,
                position="middle left",
                bold=emphasize,
            )
        )
    return traces, y_max

# visual characteristics for aggregate-value scatter point
def _aggregate_traces(action_values, chosen):
    """Dessert/resist aggregates at t=0; chosen action's value numeral is bold."""
    dessert_agg = action_values["dessert"]
    resist_agg = action_values["resist"]
    label_x = -LABEL_AGG_OFFSET

    def agg_marker(name, y, color):
        return {
            "type": "scatter",
            "mode": "markers",
            "x": [0],
            "y": [y],
            "marker": {
                "size": 12,
                "color": color,
                "opacity": 1.0,
                "line": {"color": color, "width": 1.2},
            },
            "name": f"{name} aggregate",
            "hovertemplate": f"{name} agg=%{{y:.3f}}<extra></extra>",
        }

    traces = [
        agg_marker("dessert", dessert_agg, COLORS["dessert"]),
        _value_label(
            label_x,
            dessert_agg + 0.5,
            f"{dessert_agg:.2f}",
            color=COLORS["dessert"],
            position="middle left",
            size=13,
            bold=(chosen == "dessert"),
        ),
        agg_marker("resist", resist_agg, COLORS["resist"]),
        _value_label(
            label_x,
            resist_agg + 0.5,
            f"{resist_agg:.2f}",
            color=COLORS["resist"],
            position="middle left",
            size=13,
            bold=(chosen == "resist"),
        ),
    ]
    return traces, max(dessert_agg, resist_agg)


# ---------------------------------------------------------------------------
# Dual-panel layout settings — Axis assignment, subplot shapes, hour ticks, and evenly spaced callout ys.
# ---------------------------------------------------------------------------

def _hour_xticks(x_max):
    ticks = list(range(0, x_max + 1, 12))
    labels = [str(t) for t in ticks]
    return ticks, labels


def _assign_axes(traces, *, xaxis="x", yaxis="y"):
    for tr in traces:
        tr["xaxis"] = xaxis
        tr["yaxis"] = yaxis
    return traces


def _shapes_on_subplot(shapes, *, xref="x2", yref="y2"):
    """Point sail shapes at the right-hand subplot (paper → subplot domain)."""
    out = []
    for shape in shapes:
        s = dict(shape)
        s["xref"] = xref
        if s.get("yref") == "paper":
            s["yref"] = f"{yref} domain"
        else:
            s["yref"] = yref
        out.append(s)
    return out


def _spaced_ys(count, lo, hi):
    if count <= 0:
        return []
    if count == 1:
        return [(lo + hi) / 2]
    return [lo + (hi - lo) * i / (count - 1) for i in range(count)]


# ---------------------------------------------------------------------------
# Aggregate stack + dual-panel assembler
# ---------------------------------------------------------------------------

def _hyp_stack_parts(agent, ss_events, ll_events):
    """Per-day V_hyp contributions for pure-commitment aggregate bars."""
    dessert_parts = [
        (delay, agent.value_function(reward, delay)) for delay, reward in ss_events
    ]
    resist_parts = [
        (delay, agent.value_function(reward, delay)) for delay, reward in ll_events
    ]
    return dessert_parts, resist_parts


def _sticky_stack_parts(agent, environment, state):
    """Per-day sticky EV contributions (sum to StickyHypAgent aggregates)."""
    interval = environment.decision_interval
    dessert_days = agent.choice_horizon(state, "dessert", environment)
    resist_days = agent.choice_horizon(state, "resist", environment)
    dessert_parts = [(t * interval, ev) for t, ev in dessert_days]
    resist_parts = [(t * interval, ev) for t, ev in resist_days]
    return dessert_parts, resist_parts


def _aggregate_stack_panel(dessert_parts, resist_parts, action_values, chosen):
    """Left-panel Σ bars matching visualise_sails.plot_aggregate_stack.

    dessert_parts / resist_parts are (hours, contribution) lists that sum to the
    aggregates. Returns (traces, annotations, y_peak).
    """
    n = len(dessert_parts)
    if n == 0:
        return [], [], 1.0

    dessert_agg = action_values["dessert"]
    resist_agg = action_values["resist"]

    x_dessert, x_resist = 0.0, 1.0
    bar_w = 0.55
    bottom_s = 0.0
    bottom_r = 0.0
    heights = [h for _, h in dessert_parts] + [h for _, h in resist_parts]
    min_label = 0.05 * max(heights) if heights else 0.0

    traces = []
    dessert_callouts = []
    resist_callouts = []

    for i in range(n):
        hours_s, h_s = dessert_parts[i]
        hours_r, h_r = resist_parts[i]
        edge_s = 1.2 if chosen == "dessert" else 0.5
        edge_r = 1.2 if chosen == "resist" else 0.5
        color_s = fade(COLORS["dessert_sail"], i, n)
        color_r = fade(COLORS["resist_sail"], i, n)

        traces.append(
            {
                "type": "bar",
                "x": [x_dessert],
                "y": [h_s],
                "base": [bottom_s],
                "width": [bar_w],
                "marker": {
                    "color": color_s,
                    "line": {"color": "black", "width": edge_s},
                },
                "showlegend": False,
                "hovertemplate": (
                    f"dessert @ {int(hours_s)}h<br>V=%{{y:.3f}}<extra></extra>"
                ),
            }
        )
        traces.append(
            {
                "type": "bar",
                "x": [x_resist],
                "y": [h_r],
                "base": [bottom_r],
                "width": [bar_w],
                "marker": {
                    "color": color_r,
                    "line": {"color": "black", "width": edge_r},
                },
                "showlegend": False,
                "hovertemplate": (
                    f"resist @ {int(hours_r)}h<br>V=%{{y:.3f}}<extra></extra>"
                ),
            }
        )
        if h_s >= min_label:
            dessert_callouts.append((hours_s, h_s, bottom_s + h_s / 2))
        if h_r >= min_label:
            resist_callouts.append((hours_r, h_r, bottom_r + h_r / 2))
        bottom_s += h_s
        bottom_r += h_r

    y_peak = max(dessert_agg, resist_agg) or 1.0
    annotations = []

    def callout_annotations(callouts, x_bar, x_text, *, color, xanchor):
        ys = _spaced_ys(len(callouts), y_peak * 0.06, y_peak * 0.94)
        x_edge = x_bar - bar_w / 2 if xanchor == "right" else x_bar + bar_w / 2
        for (hours, h, mid_y), y_lab in zip(callouts, ys):
            annotations.append(
                {
                    "x": x_edge,
                    "y": mid_y,
                    "ax": x_text,
                    "ay": y_lab,
                    "xref": "x",
                    "yref": "y",
                    "axref": "x",
                    "ayref": "y",
                    "text": f"{int(hours)}h<br>{h:.2f}",
                    "showarrow": True,
                    "arrowhead": 2,
                    "arrowsize": 0.9,
                    "arrowwidth": 1,
                    "arrowcolor": "#666666",
                    "font": {"size": 10, "color": color},
                    "xanchor": xanchor,
                    "align": xanchor,
                    "bgcolor": "rgba(255,255,255,0.65)",
                    "borderpad": 2,
                }
            )

    callout_annotations(
        dessert_callouts, x_dessert, x_dessert - 0.85, color=COLORS["dessert"], xanchor="right"
    )
    callout_annotations(
        resist_callouts, x_resist, x_resist + 0.85, color=COLORS["resist"], xanchor="left"
    )

    annotations.extend(
        [
            {
                "x": x_dessert - 0.75,
                "y": y_peak * 1.04,
                "xref": "x",
                "yref": "y",
                "text": f"<b>Σ dessert = {dessert_agg:.2f}</b>",
                "showarrow": False,
                "font": {"size": 12, "color": COLORS["dessert"]},
                "xanchor": "left",
                "yanchor": "bottom",
            },
            {
                "x": x_resist + 0.75,
                "y": y_peak * 1.04,
                "xref": "x",
                "yref": "y",
                "text": f"<b>Σ resist = {resist_agg:.2f}</b>",
                "showarrow": False,
                "font": {"size": 12, "color": COLORS["resist"]},
                "xanchor": "right",
                "yanchor": "bottom",
            },
        ]
    )

    _assign_axes(traces, xaxis="x", yaxis="y")
    return traces, annotations, y_peak


def _sails_dual_panel(
    agent,
    ss_events,
    ll_events,
    action,
    action_values,
    dessert_parts,
    resist_parts,
    *,
    title_prefix,
    kind="sails",
    day=None,
    extra_meta=None,
):
    """Shared dual-panel layout: aggregate stack (left) + sails (right).

    Used by sails_frame and sticky_sails_frame.

    Short vision: skip t=0 aggregate scatters/labels (they duplicate the single
    tip values) and bold the preferred option's tip / decision numeral instead.
    """
    n = len(ll_events)
    short_vision = getattr(agent, "vision", "long") == "short"
    stack_traces, stack_annotations, stack_ymax = _aggregate_stack_panel(
        dessert_parts, resist_parts, action_values, action
    )

    sail_traces = []
    shapes = []
    y_max = 0

    n_sails = len(ll_events)
    for i, (delay, reward) in enumerate(ll_events):
        decision_t = ss_events[i][0]
        traces_i, shapes_i, peak = _ll_sail_trace(
            agent,
            delay,
            reward,
            decision_t,
            sail_color=fade(COLORS["resist_sail"], i, n_sails),
            label="resist" if i == 0 else None,
            show_legend=(i == 0),
            bold_decision=short_vision and action == "resist" and i == 0,
        )
        sail_traces.extend(traces_i)
        shapes.extend(shapes_i)
        y_max = max(y_max, peak)

    ss_traces, ss_ymax = _ss_sail_traces(
        agent,
        ss_events,
        bold_first=short_vision and action == "dessert",
    )
    sail_traces.extend(ss_traces)
    y_max = max(y_max, ss_ymax)

    if not short_vision:
        agg_traces, agg_ymax = _aggregate_traces(action_values, action)
        sail_traces.extend(agg_traces)
        y_max = max(y_max, agg_ymax)

    _assign_axes(sail_traces, xaxis="x2", yaxis="y2")
    shapes = _shapes_on_subplot(shapes, xref="x2", yref="y2")

    x_max = max(
        max((d for d, _ in ll_events), default=1),
        max((d for d, _ in ss_events), default=1),
    )
    # Extra left room for aggregate labels when shown; short vision only needs tip pads
    label_pad = LABEL_DISCOUNT_OFFSET + 4 if short_vision else LABEL_AGG_OFFSET + 6
    pad_x = max(label_pad, x_max * 0.12)
    xticks, xticklabels = _hour_xticks(x_max)

    title = _choice_title(title_prefix, action)

    # Stack a bit wider than before; leave headroom above panels for subplot title
    stack_domain = [0.0, 0.3]
    sails_domain = [0.35, 1.0]
    stack_title_x = 0.5 * (stack_domain[0] + stack_domain[1])
    panel_annotations = list(stack_annotations) + [
        {
            "x": stack_title_x,
            "y": 1.0,
            "xref": "paper",
            "yref": "paper",
            "text": "Summation of discounted values<br>(Aggregate option values)",
            "showarrow": False,
            "font": {"size": 12, "color": "#1a1d23"},
            "xanchor": "center",
            "yanchor": "top",
            "align": "center",
            "yshift": -52,
        }
    ]

    meta = {
        "day": day,
        "horizon": n,
        "choice": action,
        "action_values": action_values,
        "ll_ss_gap": action_values["resist"] - action_values["dessert"],
    }
    if extra_meta:
        meta.update(extra_meta)

    return {
        "kind": kind,
        "traces": stack_traces + sail_traces,
        "layout": {
            "title": title,
            "barmode": "overlay",
            "xaxis": {
                "title": "",
                "domain": stack_domain,
                "range": [-1.25, 2.25],
                "tickvals": [0.0, 1.0],
                "ticktext": ["dessert", "resist"],
                "zeroline": False,
            },
            "yaxis": {
                "title": "Perceived value of reward",
                "domain": [0.0, 0.88],
                "range": [0, stack_ymax * 1.18],
                "gridcolor": "rgba(0,0,0,0.12)",
            },
            "xaxis2": {
                "title": "Hours from now",
                "domain": sails_domain,
                "range": [-pad_x, x_max + pad_x],
                "tickvals": xticks,
                "ticktext": xticklabels,
                "anchor": "y2",
            },
            "yaxis2": {
                "title": "Perceived value",
                "domain": [0.0, 0.88],
                "range": [0, y_max * 1.25 if y_max else 1],
                "anchor": "x2",
                "overlaying": False,
                "side": "left",
            },
            "annotations": panel_annotations,
            "shapes": shapes,
            "legend": {
                "orientation": "h",
                "y": -0.16,
                "x": sails_domain[0],
                "xref": "paper",
            },
            "margin": {"l": 60, "r": 24, "t": 72, "b": 90},
            "height": 700,
        },
        "meta": meta,
    }


# ---------------------------------------------------------------------------
# Public frame builders
# Entry points used by demo_web_app: sails, sticky sails, bundled aggregate,
# and the steppable hyperbolic approach curve.
# ---------------------------------------------------------------------------

def sails_frame(agent, environment, state, *, day=None):
    """
    Dual panel matching visualise_sails: aggregate stack (left) + sails (right).
    If day is set, title includes the day number (steppable mode).
    """
    ss_events, ll_events = foreseeable_events(agent, environment, state)
    n = len(ll_events)
    action, action_values = agent.choose_action(state, environment)
    dessert_parts, resist_parts = _hyp_stack_parts(agent, ss_events, ll_events)

    if day is not None:
        title_prefix = f"Day {day}  |  Choice sails with a horizon of {n} days"
    else:
        title_prefix = f"Choice sails with a horizon of {n} days"

    return _sails_dual_panel(
        agent,
        ss_events,
        ll_events,
        action,
        action_values,
        dessert_parts,
        resist_parts,
        title_prefix=title_prefix,
        kind="sails",
        day=day,
    )


def sticky_sails_frame(agent, environment, state):
    """
    Same dual-panel look as visualise_sails / sails_frame, but aggregates use
    StickyHypAgent perceived_s (visualise_sticky_sails / sticky_decisions).

    Light sails = pure hyperbolic SS / LL trajectories.
    Opaque t=0 scatters + left Σ bars = sticky EV bundles.
    Does not include the sticky foresight breakdown (figure 2).
    """
    ss_events, ll_events = sticky_outcome_events(agent, environment, state)
    n = len(ll_events)
    action, action_values = agent.choose_action(state, environment)
    dessert_parts, resist_parts = _sticky_stack_parts(agent, environment, state)
    perceived_s = getattr(agent, "perceived_s", None)

    return _sails_dual_panel(
        agent,
        ss_events,
        ll_events,
        action,
        action_values,
        dessert_parts,
        resist_parts,
        title_prefix=(
            f"Sticky sails | s={perceived_s} | "
            f"Choice sails with a horizon of {n} days"
        ),
        kind="sticky_sails",
        extra_meta={"perceived_s": perceived_s},
    )


def sticky_step_frame(agent, environment, state, *, day=None, last_info=None):
    """
    Stepped sticky sails + Beta posterior (sticky_step.py).

    Left: SS / LL hyperbolic sails and sticky EV aggregates under perceived s.
    Right: Beta posterior with ŝ (learn On) or fixed prior-mean (learn Off).

    learn Off → foresight uses preset perceived_s (= prior mean α₀/(α₀+β₀)).
    learn On  → foresight tracks posterior mean ŝ from stick/switch history.
    """
    beta_belief = getattr(environment, "beta_belief", None)
    if beta_belief is None:
        raise ValueError("sticky_step_frame requires environment.beta_belief")

    ss_events, ll_events = sticky_outcome_events(agent, environment, state)
    n = agent.get_horizon(state, environment)
    s_hat = environment.s_hat()
    perceived_s = agent.perceived_stickiness(environment)
    prior_mean = beta_belief.prior_mean()
    learn = bool(agent.learn_stickiness)
    intent, action_values = agent.choose_action(state, environment)

    # --- Left: sails (same geometry as sticky_step.draw_frame) ---
    sail_traces = []
    shapes = []
    y_max = 0.0

    n_sails = len(ll_events)
    for i, (delay, reward) in enumerate(ll_events):
        decision_t = ss_events[i][0]
        traces_i, shapes_i, peak = _ll_sail_trace(
            agent,
            delay,
            reward,
            decision_t,
            sail_color=fade(COLORS["resist_sail"], i, n_sails),
            label="resist" if i == 0 else None,
            show_legend=(i == 0),
        )
        sail_traces.extend(traces_i)
        shapes.extend(shapes_i)
        y_max = max(y_max, peak)

    ss_traces, ss_ymax = _ss_sail_traces(agent, ss_events)
    sail_traces.extend(ss_traces)
    y_max = max(y_max, ss_ymax)

    agg_traces, agg_ymax = _aggregate_traces(action_values, intent)
    sail_traces.extend(agg_traces)
    y_max = max(y_max, agg_ymax)

    _assign_axes(sail_traces, xaxis="x", yaxis="y")
    shapes = _shapes_on_subplot(shapes, xref="x", yref="y")

    interval = environment.decision_interval
    horizon_days = agent.horizon if agent.horizon is not None else n
    x_max = horizon_days * interval
    pad_x = max(8, x_max * 0.12)
    xticks, xticklabels = _hour_xticks(x_max)

    # --- Right: Beta posterior ---
    xs, dens = beta_belief.pdf_grid(n=251)
    y_top = max(dens) if dens else 1.0
    marker_s = perceived_s if learn else prior_mean
    marker_name = (
        f"ŝ={marker_s:.2f}" if learn else f"fixed s=α/(α+β)={prior_mean:.2f}"
    )
    belief_traces = [
        {
            "type": "scatter",
            "mode": "lines",
            "x": xs,
            "y": dens,
            "line": {"color": COLORS["belief"], "width": 2.0},
            "fill": "tozeroy",
            "fillcolor": "rgba(44,160,44,0.18)",
            "name": "Beta posterior",
            "xaxis": "x2",
            "yaxis": "y2",
            "hovertemplate": "s=%{x:.3f}<br>density=%{y:.3f}<extra></extra>",
        },
        {
            "type": "scatter",
            "mode": "lines",
            "x": [marker_s, marker_s],
            "y": [0.0, y_top * 1.15 if y_top else 1.0],
            "line": {"color": COLORS["perceived_s"], "width": 1.5},
            "name": marker_name,
            "xaxis": "x2",
            "yaxis": "y2",
            "hoverinfo": "skip",
        },
    ]

    day_bit = f"Day {day}  |  " if day is not None else ""

    last_realized = last_info.get("realized") if last_info else None
    last_stuck = last_info.get("stuck") if last_info else None
    last_intent = last_info.get("intent") if last_info else None

    # Stick/switch vs previous action (omitted until there is a prior action).
    if last_realized is None:
        vs_prior_html = ""
        vs_prior_label = None
    else:
        vs_prior = intent == last_realized
        vs_prior_label = "stick" if vs_prior else "switch"
        vs_prior_html = f" · {vs_prior_label}"

    if last_realized is None:
        last_action_html = "Last action: —"
        last_action_label = None
    elif last_stuck is None:
        last_action_html = f"Last action: <b>{last_realized}</b>"
        last_action_label = str(last_realized)
    else:
        choice_kind = "stick" if last_stuck else "switch"
        last_action_html = f"Last action: <b>{last_realized}</b> ({choice_kind})"
        last_action_label = f"{last_realized} ({choice_kind})"

    if learn:
        s_line = f"Perceived stickiness={perceived_s:.3f} (learning)"
    else:
        s_line = f"Fixed stickiness={prior_mean:.3f}"
    title = {
        "text": (
            f"{day_bit}{s_line}<br>"
            f"horizon={n}  |  <b>Agent would choose {intent}</b>{vs_prior_html}<br>"
            f"{last_action_html}"
        ),
        "font": {"size": 15, "color": "#1a1d23"},
    }

    sails_domain = [0.0, 0.66]
    belief_domain = [0.72, 1.0]
    belief_title = (
        f"Beta Distribution<br>"
        f"α={beta_belief.alpha:.1f} | β={beta_belief.beta:.1f}<br>"
        f"sticks={beta_belief.n_stick()} | switches={beta_belief.n_switch()}"
    )

    meta = {
        "day": day,
        "horizon": n,
        "choice": intent,
        "vs_prior": vs_prior_label,
        "action_values": action_values,
        "ll_ss_gap": action_values["resist"] - action_values["dessert"],
        "perceived_s": perceived_s,
        "s_hat": s_hat,
        "prior_mean": prior_mean,
        "prior_alpha": beta_belief.alpha0,
        "prior_beta": beta_belief.beta0,
        "learn_stickiness": learn,
        "alpha": beta_belief.alpha,
        "beta": beta_belief.beta,
        "n_stick": beta_belief.n_stick(),
        "n_switch": beta_belief.n_switch(),
        "last_action_label": last_action_label,
    }
    if last_info:
        meta["last_intent"] = last_intent
        meta["last_realized"] = last_realized
        meta["last_stuck"] = last_stuck

    return {
        "kind": "sticky_step",
        "traces": sail_traces + belief_traces,
        "layout": {
            "title": title,
            "xaxis": {
                "title": "Hours from now",
                "domain": sails_domain,
                "range": [-pad_x, x_max + pad_x],
                "tickvals": xticks,
                "ticktext": xticklabels,
                "anchor": "y",
            },
            "yaxis": {
                "title": "Perceived value",
                "domain": [0.0, 0.88],
                "range": [0, y_max * 1.25 if y_max else 1],
                "anchor": "x",
            },
            "xaxis2": {
                "title": "Stickiness s",
                "domain": belief_domain,
                "range": [0.0, 1.0],
                "anchor": "y2",
            },
            "yaxis2": {
                "title": "Density",
                "domain": [0.0, 0.88],
                "range": [0.0, y_top * 1.15 if y_top else 1.0],
                "anchor": "x2",
            },
            "annotations": [
                {
                    "x": 0.5 * (belief_domain[0] + belief_domain[1]),
                    "y": 1.0,
                    "xref": "paper",
                    "yref": "paper",
                    "text": belief_title,
                    "showarrow": False,
                    "font": {"size": 12, "color": "#1a1d23"},
                    "xanchor": "center",
                    "yanchor": "top",
                    "align": "center",
                    "yshift": -8,
                }
            ],
            "shapes": shapes,
            "legend": {
                "orientation": "h",
                "y": -0.16,
                "x": 0.0,
                "xref": "paper",
            },
            "margin": {"l": 60, "r": 24, "t": 92, "b": 90},
            "height": 700,
        },
        "meta": meta,
    }


def aggregate_frame(agent, environment, state):
    """Bundled resist sail + dessert aggregate, matching aggregate_sail.py."""
    action, action_values = agent.choose_action(state, environment)
    ss_events = agent.choice_horizon(state, "dessert", environment)
    ll_events = agent.choice_horizon(state, "resist", environment)
    ll_delay = float(environment.rewards["LL"]["delay"])

    def bundle_value_at(events, at_time):
        return sum(
            agent.value_function(reward, delay - at_time)
            for delay, reward in events
            if delay >= at_time
        )

    if ll_delay <= 0:
        xs, ys = [0.0], [float(bundle_value_at(ll_events, 0))]
    else:
        n = SAIL_SAMPLE_POINTS
        xs = [ll_delay * i / (n - 1) for i in range(n)]
        ys = [float(bundle_value_at(ll_events, t)) for t in xs]
    ss_value = bundle_value_at(ss_events, 0)
    perceived_ll = bundle_value_at(ll_events, 0)
    ll_tip = bundle_value_at(ll_events, ll_delay)
    y_max = max(max(ys) if ys else 0, ss_value, perceived_ll, ll_tip)

    resist_chosen = action == "resist"
    dessert_chosen = action == "dessert"
    resist_w = 3.5 if resist_chosen else 2.0
    dessert_size = 16 if dessert_chosen else 10
    resist_size = 14 if resist_chosen else 10

    traces = [
        {
            "type": "scatter",
            "mode": "lines",
            "x": xs,
            "y": ys,
            "line": {
                "color": COLORS["resist"],
                "width": resist_w,
            },
            "opacity": 1.0,
            "name": "resist" + (" ★" if resist_chosen else ""),
            "hovertemplate": "t=%{x}h<br>V=%{y:.3f}<extra></extra>",
        },
        {
            "type": "scatter",
            "mode": "markers",
            "x": [0],
            "y": [perceived_ll],
            "marker": {
                "size": resist_size,
                "color": COLORS["resist"],
                "opacity": 1.0,
                "line": {"color": COLORS["resist"], "width": 2 if resist_chosen else 0},
            },
            "showlegend": False,
            "hoverinfo": "skip",
        },
        _value_label(
            0,
            perceived_ll,
            f"{perceived_ll:.2f}",
            color=COLORS["resist"],
            position="top left",
        ),
        {
            "type": "scatter",
            "mode": "markers",
            "x": [ll_delay],
            "y": [ll_tip],
            "marker": {
                "size": resist_size,
                "color": COLORS["resist"],
                "opacity": 1.0,
            },
            "showlegend": False,
            "hoverinfo": "skip",
        },
        _value_label(
            ll_delay,
            ll_tip,
            f"{ll_tip:.2f}",
            color=COLORS["resist"],
            position="top right",
        ),
        {
            "type": "scatter",
            "mode": "markers",
            "x": [0],
            "y": [ss_value],
            "marker": {
                "size": dessert_size,
                "color": COLORS["dessert"],
                "opacity": 1.0,
                "line": {"color": COLORS["dessert"], "width": 2.5 if dessert_chosen else 0},
            },
            "name": "dessert" + (" ★" if dessert_chosen else ""),
            "hovertemplate": "dessert=%{y:.3f}<extra></extra>",
        },
        _value_label(
            0,
            ss_value,
            f"{ss_value:.2f}",
            color=COLORS["dessert"],
            position="bottom left",
        ),
    ]
    # Bold font on the chosen action's value numeral(s)
    if resist_chosen:
        traces[2]["textfont"] = {
            "color": COLORS["resist"],
            "size": 14,
            "family": "Arial Black, Helvetica Neue, sans-serif",
        }
    if dessert_chosen:
        traces[-1]["textfont"] = {
            "color": COLORS["dessert"],
            "size": 14,
            "family": "Arial Black, Helvetica Neue, sans-serif",
        }

    shapes = [
        {
            "type": "line",
            "x0": ll_delay,
            "x1": ll_delay,
            "y0": 0,
            "y1": 1,
            "yref": "paper",
            "line": {"color": COLORS["resist"], "width": 1, "dash": "dot"},
            "opacity": 0.5,
        }
    ]
    pad = max(2, ll_delay * 0.2)

    return {
        "kind": "aggregate_sail",
        "traces": traces,
        "layout": {
            "title": _choice_title("Bundled value on day 1", action),
            "xaxis": {
                "title": "Hours from now",
                "range": [-pad, ll_delay + max(1, ll_delay * 0.15)],
            },
            "yaxis": {
                "title": "Perceived value",
                "range": [0, y_max * 1.15 if y_max else 1],
            },
            "shapes": shapes,
            "legend": {"orientation": "h", "y": -0.15},
            "margin": {"l": 60, "r": 20, "t": 50, "b": 70},
            "height": 620,
        },
        "meta": {
            "choice": action,
            "action_values": action_values,
        },
    }


def hyperbolic_frame(reward, initial_delay, discount_factor, *, elapsed=0):
    """
    Reward approaching an agent fixed at t=0 (hyperbolic_curve.py motion).

    ``elapsed`` is how far time has advanced toward delivery (0 … initial_delay).
    Perceived values use HypAgent maths: V = reward / (1 + k * distance).

    Axis ranges stay locked to the initial delay and reward so the plot does
    not rescale as the sail shortens. Sail traces are densely sampled for a
    smooth curve; stepping still advances ``elapsed`` by whole hours.
    """
    initial_delay = int(initial_delay)
    elapsed = max(0, min(int(elapsed), initial_delay))
    remaining = initial_delay - elapsed
    k = float(discount_factor)

    def V(distance):
        return reward / (1.0 + k * distance)

    xs, ys = _dense_hyperbolic_xy(reward, remaining, k)
    perceived = V(remaining)

    # Fixed scales for the whole approach (match hyperbolic_curve.py)
    x_hi = max(initial_delay, 1)
    y_hi = float(reward) + 1.0

    traces = [
        {
            "type": "scatter",
            "mode": "lines",
            "x": xs,
            "y": ys,
            "line": {"color": COLORS["resist"], "width": 2.5},
            "name": "hyperbolic sail",
            "hovertemplate": "t=%{x}<br>V=%{y:.3f}<extra></extra>",
        },
        {
            "type": "scatter",
            "mode": "markers+text",
            "x": [0],
            "y": [perceived],
            "text": [f"{perceived:.2f}"],
            "textposition": "top center",
            "textfont": {"color": "#1a1d23", "size": 13},
            "marker": {"size": 12, "color": "black", "opacity": 1.0},
            "name": "perceived at day 0",
            "hovertemplate": "perceived=%{y:.3f}<extra></extra>",
        },
    ]
    shapes = [
        {
            "type": "line",
            "x0": remaining,
            "x1": remaining,
            "y0": 0,
            "y1": 1,
            "yref": "paper",
            "line": {"color": "#888", "width": 1, "dash": "dash"},
        }
    ]

    return {
        "kind": "hyperbolic_curve",
        "traces": traces,
        "layout": {
            "title": {
                "text": (
                    f"Reward approaching | Time elapsed = {elapsed}h  |  "
                    f"remaining = {remaining}h  |  "
                    f"<b>V(0) = {perceived:.2f}</b>"
                ),
                "font": {"size": 15, "color": "#1a1d23"},
            },
            "xaxis": {
                "title": "Hours to approaching reward",
                "range": [-0.2, x_hi],
                "autorange": False,
                "fixedrange": True,
            },
            "yaxis": {
                "title": "Perceived value",
                "range": [0, y_hi],
                "autorange": False,
                "fixedrange": True,
            },
            "shapes": shapes,
            "legend": {"orientation": "h", "y": -0.15},
            "margin": {"l": 60, "r": 20, "t": 50, "b": 70},
            "height": 620,
            # Keep zoom/axis identity stable across steps for this param set
            "uirevision": f"hyp|{reward}|{initial_delay}|{k}",
        },
        "meta": {
            "reward": reward,
            "initial_delay": initial_delay,
            "elapsed": elapsed,
            "remaining": remaining,
            "discount_factor": k,
            "perceived": perceived,
            "finished": elapsed >= initial_delay,
        },
    }


def ss_delay_from_ll(ll_delay):
    """SS delay = ¾ of LL delay, rounded to the nearest hour (≥ 0)."""
    return max(0, int(round(0.75 * int(ll_delay))))


def hyperbolic_ss_ll_frame(
    ss_value, ll_value, ll_delay, discount_factor, *, elapsed=0
):
    """
    SS and LL rewards approaching an agent fixed at t=0.

    Same motion as ``hyperbolic_frame``, but both outcomes share one timeline.
    LL value/delay and SS value come from the shared panel; SS delay is derived
    as ¾ × LL delay (rounded to nearest hour). Perceived values use
    V = reward / (1 + k * remaining). The preferred option is emphasised.
    """
    ss_value = float(ss_value)
    ll_value = float(ll_value)
    ll_delay = int(ll_delay)
    ss_delay = ss_delay_from_ll(ll_delay)
    k = float(discount_factor)
    max_delay = max(ss_delay, ll_delay)
    elapsed = max(0, min(int(elapsed), max_delay))

    def V(reward, distance):
        return reward / (1.0 + k * distance)

    rem_ss = max(0, ss_delay - elapsed)
    rem_ll = max(0, ll_delay - elapsed)
    # Still choosable at the delivery instant (remaining == 0); past that → delivered.
    ss_available = elapsed <= ss_delay
    ll_available = elapsed <= ll_delay
    v_ss = V(ss_value, rem_ss) if ss_available else None
    v_ll = V(ll_value, rem_ll) if ll_available else None

    if ss_available and ll_available:
        if abs(v_ss - v_ll) < 1e-12:
            preference = "indifferent"
        elif v_ss > v_ll:
            preference = "SS"
        else:
            preference = "LL"
    elif ss_available:
        preference = "SS"
    elif ll_available:
        preference = "LL"
    else:
        preference = "none"

    xs_ss, ys_ss = (
        _dense_hyperbolic_xy(ss_value, rem_ss, k)
        if ss_available
        else ([0.0], [ss_value])
    )
    xs_ll, ys_ll = (
        _dense_hyperbolic_xy(ll_value, rem_ll, k)
        if ll_available
        else ([0.0], [ll_value])
    )

    x_hi = max(max_delay, 1)
    y_hi = max(ss_value, ll_value) + 1.0

    def option_style(name, available):
        if not available:
            return {
                "width": 1.2,
                "marker": 7,
                "opacity": 0.2,
                "dash": "dot",
                "suffix": " (delivered)",
                "chosen": False,
            }
        # Indifferent: both get emphasis so the tie is visible.
        chosen = preference == "indifferent" or preference == name
        return {
            "width": 3.6 if chosen else 1.8,
            "marker": 14 if chosen else 9,
            "opacity": 1.0 if chosen else 0.45,
            "dash": "solid",
            "suffix": (
                " ="
                if preference == "indifferent"
                else (" ★" if chosen else "")
            ),
            "chosen": chosen,
        }

    ss_style = option_style("SS", ss_available)
    ll_style = option_style("LL", ll_available)

    def now_label(name, value, available):
        if not available:
            return f"{name} delivered"
        return f"{name} {value:.2f}"

    traces = [
        {
            "type": "scatter",
            "mode": "lines",
            "x": xs_ss,
            "y": ys_ss,
            "line": {
                "color": COLORS["dessert"],
                "width": ss_style["width"],
                "dash": ss_style["dash"],
            },
            "opacity": ss_style["opacity"],
            "name": f"SS sail{ss_style['suffix']}",
            "hovertemplate": "SS t=%{x}<br>V=%{y:.3f}<extra></extra>",
        },
        {
            "type": "scatter",
            "mode": "markers+text",
            "x": [0],
            "y": [v_ss if ss_available else ss_value],
            "text": [
                now_label("SS", v_ss if ss_available else ss_value, ss_available)
            ],
            "textposition": "top center",
            "textfont": {
                "color": COLORS["dessert"],
                "size": 14 if preference == "SS" else 12,
                "family": (
                    "Arial Black, sans-serif"
                    if preference == "SS"
                    else "Arial, sans-serif"
                ),
            },
            "marker": {
                "size": ss_style["marker"],
                "color": COLORS["dessert"],
                "opacity": ss_style["opacity"],
                "line": {
                    "width": 2 if preference == "SS" else 0,
                    "color": "#1a1d23",
                },
            },
            "name": f"SS at now{ss_style['suffix']}",
            "hovertemplate": (
                f"SS V(0)={v_ss:.3f}<extra></extra>"
                if ss_available
                else "SS delivered<extra></extra>"
            ),
            "showlegend": False,
        },
        {
            "type": "scatter",
            "mode": "lines",
            "x": xs_ll,
            "y": ys_ll,
            "line": {
                "color": COLORS["resist"],
                "width": ll_style["width"],
                "dash": ll_style["dash"],
            },
            "opacity": ll_style["opacity"],
            "name": f"LL sail{ll_style['suffix']}",
            "hovertemplate": "LL t=%{x}<br>V=%{y:.3f}<extra></extra>",
        },
        {
            "type": "scatter",
            "mode": "markers+text",
            "x": [0],
            "y": [v_ll if ll_available else ll_value],
            "text": [
                now_label("LL", v_ll if ll_available else ll_value, ll_available)
            ],
            "textposition": (
                "bottom center"
                if (
                    ss_available
                    and ll_available
                    and abs(v_ss - v_ll) < 0.35
                )
                else "top center"
            ),
            "textfont": {
                "color": COLORS["resist"],
                "size": 14 if preference == "LL" else 12,
                "family": (
                    "Arial Black, sans-serif"
                    if preference == "LL"
                    else "Arial, sans-serif"
                ),
            },
            "marker": {
                "size": ll_style["marker"],
                "color": COLORS["resist"],
                "opacity": ll_style["opacity"],
                "line": {
                    "width": 2 if preference == "LL" else 0,
                    "color": "#1a1d23",
                },
            },
            "name": f"LL at now{ll_style['suffix']}",
            "hovertemplate": (
                f"LL V(0)={v_ll:.3f}<extra></extra>"
                if ll_available
                else "LL delivered<extra></extra>"
            ),
            "showlegend": False,
        },
    ]

    shapes = []
    if ss_available:
        shapes.append(
            {
                "type": "line",
                "x0": rem_ss,
                "x1": rem_ss,
                "y0": 0,
                "y1": 1,
                "yref": "paper",
                "line": {"color": COLORS["dessert"], "width": 1, "dash": "dash"},
                "opacity": 0.55,
            }
        )
    if ll_available:
        shapes.append(
            {
                "type": "line",
                "x0": rem_ll,
                "x1": rem_ll,
                "y0": 0,
                "y1": 1,
                "yref": "paper",
                "line": {"color": COLORS["resist"], "width": 1, "dash": "dash"},
                "opacity": 0.55,
            }
        )

    if preference == "indifferent":
        pref_html = "<b>indifferent</b>"
    elif preference == "none":
        pref_html = "<b>both delivered</b>"
    else:
        pref_html = f"<b>prefers {preference}</b>"


    return {
        "kind": "hyperbolic_ss_ll",
        "traces": traces,
        "layout": {
            "title": {
                "text": (
                    f"Rewards approaching | Time elapsed = {elapsed}h  |  "
                    f"SS rem = {rem_ss}h  ·  LL rem = {rem_ll}h  |  "
                    f"{pref_html}"
                ),
                "font": {"size": 15, "color": "#1a1d23"},
            },
            "xaxis": {
                "title": "Hours to approaching reward",
                "range": [-0.2, x_hi],
                "autorange": False,
                "fixedrange": True,
            },
            "yaxis": {
                "title": "Perceived value",
                "range": [0, y_hi],
                "autorange": False,
                "fixedrange": True,
            },
            "shapes": shapes,
            "legend": {"orientation": "h", "y": -0.15},
            "margin": {"l": 60, "r": 20, "t": 70, "b": 70},
            "height": 620,
            "uirevision": (
                f"hyp_ss_ll|{ss_value}|{ss_delay}|{ll_value}|{ll_delay}|{k}"
            ),
        },
        "meta": {
            "ss_value": ss_value,
            "ss_delay": ss_delay,
            "ll_value": ll_value,
            "ll_delay": ll_delay,
            "discount_factor": k,
            "elapsed": elapsed,
            "remaining_ss": rem_ss,
            "remaining_ll": rem_ll,
            "ss_available": ss_available,
            "ll_available": ll_available,
            "v_ss": v_ss,
            "v_ll": v_ll,
            "preference": preference,
            "finished": elapsed >= max_delay,
            "ss_delay_rule": "0.75 * ll_delay (rounded)",
        },
    }


# ---------------------------------------------------------------------------
# Sticky max-SS-resisted curves
# Sweep perceived stickiness × LL delay; uses max_ss_resisted / sticky_delay_rgba above.
# ---------------------------------------------------------------------------

def stickiness_axis():
    step = (STICKINESS_MAX - STICKINESS_MIN) / (N_STICKINESS_POINTS - 1)
    return [STICKINESS_MIN + i * step for i in range(N_STICKINESS_POINTS)]


def delay_list(delay_max, delay_step):
    delay_max = max(0, int(delay_max))
    delay_step = max(1, int(delay_step))
    return list(range(0, delay_max + 1, delay_step))


def sticky_curves(agent, environment, state, delay_max, delay_step, reward_LL=None):
    """Family of max-SS-resisted curves; same model as sticky_graph.py."""
    horizon = agent.get_horizon(state, environment)
    decision_interval = environment.decision_interval
    delay_SS = environment.rewards["SS"]["delay"]
    current_delay_LL = int(environment.rewards["LL"]["delay"])
    if reward_LL is None:
        reward_LL = environment.rewards["LL"]["value"]

    delays = delay_list(delay_max, delay_step)
    # Ensure the configured LL delay is present so it can be highlighted
    if current_delay_LL not in delays:
        delays = sorted(delays + [current_delay_LL])
    xs = stickiness_axis()
    curves = []
    for d in delays:
        max_ss = []
        for s in xs:
            try:
                val = max_ss_resisted(
                    agent, s, delay_SS, d, reward_LL, horizon, decision_interval
                )
            except ZeroDivisionError:
                val = None
            max_ss.append(val if val is not None and abs(val) != float("inf") else None)
        curves.append({"delay_LL": d, "max_ss": max_ss})

    y_peak = 0
    for c in curves:
        for v in c["max_ss"]:
            if v is not None and v > y_peak:
                y_peak = v
    ymax = max(8, y_peak * 1.15) if y_peak else 8

    # Default y zoom: one delay-step of context around the current (black) curve
    focus_delays = {current_delay_LL}
    if current_delay_LL in delays:
        idx = delays.index(current_delay_LL)
        if idx > 0:
            focus_delays.add(delays[idx - 1])
        if idx < len(delays) - 1:
            focus_delays.add(delays[idx + 1])
    focus_vals = [
        v
        for c in curves
        if c["delay_LL"] in focus_delays
        for v in c["max_ss"]
        if v is not None
    ]
    # Default y zoom: center on nearby delays, but keep a tall enough window
    # so the initial view is not a short band around the focus curves.
    if focus_vals:
        lo, hi = min(focus_vals), max(focus_vals)
        pad = max((hi - lo) * 0.45, 1.5)
        y_lo = max(0.0, lo - pad)
        y_hi = hi + pad
        min_span = max(ymax * 0.55, 4.0)
        if y_hi - y_lo < min_span:
            mid = 0.5 * (y_lo + y_hi)
            y_lo = max(0.0, mid - 0.5 * min_span)
            y_hi = y_lo + min_span
        if y_hi <= y_lo:
            y_hi = y_lo + 1.0
    else:
        y_lo, y_hi = 0.0, ymax

    delay_lo = min(delays) if delays else 0
    delay_hi = max(delays) if delays else 1
    traces = []
    current_trace_index = None
    for i, c in enumerate(curves):
        is_current = c["delay_LL"] == current_delay_LL
        if is_current:
            current_trace_index = i
        traces.append(
            {
                "type": "scatter",
                "mode": "lines",
                "x": xs,
                "y": c["max_ss"],
                "line": {
                    "width": 3.2 if is_current else 2.5,
                    "color": (
                        "#000000"
                        if is_current
                        else sticky_delay_rgba(c["delay_LL"], delay_lo, delay_hi)
                    ),
                },
                "name": (
                    f"LL delay = {c['delay_LL']}h (current)"
                    if is_current
                    else f"LL delay = {c['delay_LL']}h"
                ),
                "hovertemplate": (
                    "s=%{x:.2f}<br>max SS=%{y:.3f}"
                    + (
                        "<br><i>Click → set sticky sails s</i>"
                        if is_current
                        else ""
                    )
                    + "<extra></extra>"
                ),
                "meta": {"is_current": is_current},
            }
        )

    return {
        "kind": "sticky",
        "traces": traces,
        "layout": {
            "title": (
                f"Max SS resisted | LL={reward_LL}, current delay={current_delay_LL}h, "
                f"horizon={horizon}, discount_factor={agent.discount_factor}"
            ),
            "xaxis": {
                "title": "Perceived Stickiness (s)",
                "range": [0, 1],
                "autorange": False,
                # Lock s ∈ [0, 1]: pan/zoom can still move y, but not past perceived-stickiness bounds
                "fixedrange": True,
            },
            "yaxis": {
                "title": "Max SS resisted",
                "range": [y_lo, y_hi],
                "autorange": False,
            },
            "legend": {"orientation": "h", "y": -0.22},
            "margin": {"l": 60, "r": 20, "t": 50, "b": 80},
            "height": 420,
            "hovermode": "closest",
        },
        "meta": {
            "stickiness": xs,
            "delays": delays,
            "current_delay_LL": current_delay_LL,
            "current_trace_index": current_trace_index,
            "reward_LL": reward_LL,
            "horizon": horizon,
            "discount_factor": agent.discount_factor,
            "y_range": [y_lo, y_hi],
        },
    }


# ---------------------------------------------------------------------------
# Sticky learning scan (scan_sticky_learning.py) — Plotly twin of matplotlib
# ---------------------------------------------------------------------------

_SCAN_LABEL_COLORS = {
    "lock_in_rising_s": "#2ca02c",
    "lock_in": "#7f7f7f",
    "switching_loop": "#d62728",
}


def _scan_row_color(row):
    """One-shot markers use the action switched to; else regime colour."""
    if row.get("label") == "one_shot_switch":
        last = row.get("last")
        if last == "dessert":
            return COLORS["dessert"]
        if last == "resist":
            return COLORS["resist"]
    return _SCAN_LABEL_COLORS.get(row.get("label"), "#cccccc")


def _scan_equal_row_edges(n_rows):
    """Equal-height band edges for n displayed rows on an index y-axis."""
    n = max(0, int(n_rows))
    if n == 0:
        return []
    return [i - 0.5 for i in range(n + 1)]


def _beta_prior_panel_traces(alpha, beta, *, mean_s=None):
    """Prior Beta(α, β) density + mean marker for the scan right panel."""
    belief = BayesianStickinessBelief(alpha=float(alpha), beta=float(beta))
    xs, dens = belief.pdf_grid(n=251)
    y_top = max(dens) if dens else 1.0
    marker_s = float(mean_s) if mean_s is not None else belief.mean()
    traces = [
        {
            "type": "scatter",
            "mode": "lines",
            "x": xs,
            "y": dens,
            "line": {"color": COLORS["belief"], "width": 2.0},
            "fill": "tozeroy",
            "fillcolor": "rgba(44,160,44,0.18)",
            "name": "prior Beta",
            "xaxis": "x2",
            "yaxis": "y2",
            "hovertemplate": "s=%{x:.3f}<br>density=%{y:.3f}<extra></extra>",
            "showlegend": False,
        },
        {
            "type": "scatter",
            "mode": "lines",
            "x": [marker_s, marker_s],
            "y": [0.0, y_top * 1.15 if y_top else 1.0],
            "line": {"color": COLORS["perceived_s"], "width": 1.5},
            "name": f"mean s={marker_s:.3f}",
            "xaxis": "x2",
            "yaxis": "y2",
            "hoverinfo": "skip",
            "showlegend": False,
        },
    ]
    return traces, y_top, marker_s


def _resolve_selected_row(rows, selected_s=None, selected_index=None):
    """Pick a scan row by index or prior mean s (exact within float noise)."""
    if not rows:
        return None, None
    if selected_index is not None:
        try:
            idx = int(selected_index)
        except (TypeError, ValueError):
            idx = None
        if idx is not None and 0 <= idx < len(rows):
            return idx, rows[idx]
    if selected_s is None:
        return None, None
    try:
        target = float(selected_s)
    except (TypeError, ValueError):
        return None, None
    best_i = min(
        range(len(rows)),
        key=lambda i: abs(float(rows[i]["s"]) - target),
    )
    if abs(float(rows[best_i]["s"]) - target) <= 1e-6:
        return best_i, rows[best_i]
    return None, None


def scan_frame(
    rows,
    *,
    n_steps,
    strength_min=None,
    strength_max=None,
    selected_s=None,
    selected_index=None,
):
    """
    Action heatmap (day × prior s) + prior Beta PDF for a clicked row.
    rows: output of scan_sticky_learning.build_scan.

    strength_min / strength_max filter the view by prior N=α+β only
    (priors themselves stay lowest integers for each mean).
    """
    if strength_min is not None or strength_max is not None:
        lo = 0 if strength_min is None else int(strength_min)
        hi = 10**9 if strength_max is None else int(strength_max)
        if lo > hi:
            lo, hi = hi, lo
        rows = [
            r
            for r in rows
            if lo <= int(r.get("n", int(r["alpha"]) + int(r["beta"]))) <= hi
        ]

    if not rows:
        return {
            "kind": "sticky_scan",
            "traces": [],
            "layout": {
                "title": {
                    "text": "Sticky learning scan (no rows in strength filter)"
                }
            },
            "meta": {"n_steps": n_steps, "n_points": 0, "rows": []},
        }

    s_vals = [float(r["s"]) for r in rows]
    n_rows = len(rows)
    # Index y-axis → equal row heights regardless of gaps in prior mean s.
    y_idx = list(range(n_rows))
    z = [[0 if a == "dessert" else 1 for a in r["actions"]] for r in rows]
    sel_i, sel_row = _resolve_selected_row(
        rows, selected_s=selected_s, selected_index=selected_index
    )

    # Per-cell hover: [action, alpha, beta, N, row_index, s]
    custom = []
    for i, r in enumerate(rows):
        a = int(r["alpha"])
        b = int(r["beta"])
        n = int(r.get("n", a + b))
        s_i = float(r["s"])
        custom.append(
            [
                [
                    "dessert" if v == 0 else "resist",
                    a,
                    b,
                    n,
                    i,
                    s_i,
                ]
                for v in z[i]
            ]
        )

    edges = _scan_equal_row_edges(n_rows)
    y_lo, y_hi = float(edges[0]), float(edges[-1])

    heatmap = {
        "type": "heatmap",
        "x": list(range(1, n_steps + 1)),
        "y": y_idx,
        "z": z,
        "colorscale": [
            [0.0, COLORS["dessert"]],
            [1.0, COLORS["resist"]],
        ],
        "zmin": 0,
        "zmax": 1,
        "showscale": False,
        "xgap": 0,
        "ygap": 1,
        "hovertemplate": (
            "day=%{x}<br>s=%{customdata[5]:.3f}<br>"
            "α=%{customdata[1]} · β=%{customdata[2]} · N=%{customdata[3]}<br>"
            "action=%{customdata[0]}<extra></extra>"
        ),
        "customdata": custom,
        "xaxis": "x",
        "yaxis": "y",
        "name": "action",
    }

    # Horizontal separators at every equal-height prior-s band edge.
    shapes = [
        {
            "type": "line",
            "xref": "x",
            "yref": "y",
            "x0": 0.5,
            "x1": n_steps + 0.5,
            "y0": float(y_edge),
            "y1": float(y_edge),
            "line": {"color": "rgba(255,255,255,0.92)", "width": 1.25},
            "layer": "above",
        }
        for y_edge in edges
    ]
    annotations = [
        {
            "text": "red=dessert · blue=resist · click a row for prior Beta",
            "xref": "paper",
            "yref": "paper",
            "x": 0.36,
            "y": 1.08,
            "showarrow": False,
            "font": {"size": 11, "color": "#5c6570"},
        }
    ]

    if sel_row is not None:
        alpha = int(sel_row["alpha"])
        beta = int(sel_row["beta"])
        n_prior = int(sel_row.get("n", alpha + beta))
        mean_s = float(sel_row["s"])
        prior_traces, y_top, marker_s = _beta_prior_panel_traces(
            alpha, beta, mean_s=mean_s
        )
        belief_title = (
            f"Prior Beta(α, β)<br>"
            f"α={alpha} · β={beta} · N={n_prior}<br>"
            f"mean s={mean_s:.4f}"
        )
        shapes.append(
            {
                "type": "rect",
                "xref": "x",
                "yref": "y",
                "x0": 0.5,
                "x1": n_steps + 0.5,
                "y0": float(edges[sel_i]),
                "y1": float(edges[sel_i + 1]),
                "fillcolor": "rgba(0,0,0,0.10)",
                "line": {
                    "color": "rgba(26,29,35,0.85)",
                    "width": 1.5,
                },
                "layer": "above",
            }
        )
    else:
        prior_traces = [
            {
                "type": "scatter",
                "mode": "lines",
                "x": [],
                "y": [],
                "xaxis": "x2",
                "yaxis": "y2",
                "showlegend": False,
                "hoverinfo": "skip",
            }
        ]
        y_top = 1.0
        marker_s = None
        belief_title = "Click a heatmap row<br>to show prior Beta(α, β)"

    annotations.append(
        {
            "x": 0.90,
            "y": 1.0,
            "xref": "paper",
            "yref": "paper",
            "text": belief_title,
            "showarrow": False,
            "font": {"size": 12, "color": "#1a1d23"},
            "xanchor": "center",
            "yanchor": "top",
            "align": "center",
            "yshift": -4,
        }
    )

    # Label every displayed row by its prior mean; thin out if crowded.
    tick_step = 1 if n_rows <= 40 else max(1, int(round(n_rows / 30)))
    tickvals = y_idx[::tick_step]
    ticktext = [f"{s_vals[i]:.3f}" for i in tickvals]

    row_meta = [
        {
            "s": float(r["s"]),
            "alpha": int(r["alpha"]),
            "beta": int(r["beta"]),
            "n": int(r.get("n", int(r["alpha"]) + int(r["beta"]))),
            "label": r.get("label"),
            "detail": r.get("detail", ""),
        }
        for r in rows
    ]

    return {
        "kind": "sticky_scan",
        "traces": [heatmap] + prior_traces,
        "layout": {
            "title": {
                "text": (
                    f"Sticky learning scan · steps={n_steps} · "
                    f"points={len(rows)}"
                    + (
                        f" · N∈[{strength_min},{strength_max}]"
                        if strength_min is not None or strength_max is not None
                        else ""
                    )
                ),
                "font": {"size": 15, "color": "#1a1d23"},
            },
            "xaxis": {
                "title": "Decision day",
                "domain": [0.0, 0.66],
                "range": [0.5, n_steps + 0.5],
                "anchor": "y",
            },
            "yaxis": {
                "title": "Prior mean s",
                "domain": [0.0, 0.92],
                "range": [y_lo, y_hi],
                "tickvals": tickvals,
                "ticktext": ticktext,
                "anchor": "x",
            },
            "xaxis2": {
                "title": "Stickiness s",
                "domain": [0.74, 1.0],
                "range": [0.0, 1.0],
                "anchor": "y2",
            },
            "yaxis2": {
                "title": "Density",
                "domain": [0.0, 0.92],
                "range": [0.0, y_top * 1.15 if y_top else 1.0],
                "anchor": "x2",
            },
            "annotations": annotations,
            "shapes": shapes,
            "margin": {"l": 70, "r": 30, "t": 78, "b": 55},
            "height": 560,
            "showlegend": False,
        },
        "meta": {
            "n_steps": n_steps,
            "n_points": len(rows),
            "s_min": s_vals[0],
            "s_max": s_vals[-1],
            "strength_min": (
                min(int(r.get("n", r["alpha"] + r["beta"])) for r in rows)
            ),
            "strength_max": (
                max(int(r.get("n", r["alpha"] + r["beta"])) for r in rows)
            ),
            "labels": [r["label"] for r in rows],
            "rows": row_meta,
            "selected_index": sel_i,
            "selected_s": None if sel_row is None else float(sel_row["s"]),
            "selected_alpha": None if sel_row is None else int(sel_row["alpha"]),
            "selected_beta": None if sel_row is None else int(sel_row["beta"]),
            "selected_n": (
                None
                if sel_row is None
                else int(sel_row.get("n", sel_row["alpha"] + sel_row["beta"]))
            ),
            "selected_mean": marker_s,
        },
    }
