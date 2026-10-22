"""
JSON series builders for the unified demo web app.

Pure helpers: turn agent/env state into Plotly-ready payloads. No matplotlib.
Math matches the corresponding visual_demos/*.py scripts.

Sticky solve mirrors sticky_graph.max_ss_resisted (duplicated here so importing
this module does not run sticky_graph's module-level matplotlib script).
"""

from sticky_decisions import stick_p_t

COLORS = {
    "snack": "#c44e52",
    "resist": "#4c72b0",
    "snack_agg": "#e89a9e",
    "resist_agg": "#8fb0d9",
}

# Hours left of the marker for left-side numerals.
# Aggregates sit further left than per-outcome discounted values (matplotlib: ~-2 vs decision-node labels).
LABEL_DISCOUNT_OFFSET = 8
LABEL_AGG_OFFSET = 20

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
    V_snack_LL = 0.0
    V_snack_SS = 0.0

    for t in range(horizon):
        sticking = stick_p_t(s, t)
        v_SS = agent.value_function(1.0, delay_SS + t * decision_interval)
        v_LL = agent.value_function(1.0, delay_LL + t * decision_interval)

        V_resist_LL += sticking * v_LL
        V_resist_SS += (1.0 - sticking) * v_SS
        V_snack_SS += sticking * v_SS
        V_snack_LL += (1.0 - sticking) * v_LL

    denom = V_snack_SS - V_resist_SS
    if denom == 0:
        return None
    return reward_LL * (V_resist_LL - V_snack_LL) / denom


def foreseeable_events(agent, environment, state):
    return (
        agent.choice_horizon(state, "snack", environment),
        agent.choice_horizon(state, "resist", environment),
    )


def _value_label(x, y, text, *, color, position="middle left"):
    """Numeral only (no marker). Colour matches matplotlib text on that series."""
    return {
        "type": "scatter",
        "mode": "text",
        "x": [x],
        "y": [y],
        "text": [text],
        "textposition": position,
        "textfont": {"color": color, "size": 12},
        "showlegend": False,
        "hoverinfo": "skip",
    }


def _ll_sail_trace(agent, delay, reward, decision_t, *, label=None, show_legend=True):
    """One LL hyperbolic sail + tip / decision-node markers."""
    xs = list(range(0, delay + 1))
    ys = [agent.value_function(reward, delay - t) for t in xs]
    tip = ys[-1] if ys else 0
    decision_val = agent.value_function(reward, delay - decision_t)
    # Discounted value at decision node — left of marker, but right of aggregates
    decision_label_x = decision_t - LABEL_DISCOUNT_OFFSET

    traces = [
        {
            "type": "scatter",
            "mode": "lines",
            "x": xs,
            "y": ys,
            "line": {"color": COLORS["resist"], "width": 2},
            "name": label or "resist",
            "showlegend": show_legend and label is not None,
            "hovertemplate": "t=%{x}h<br>V=%{y:.3f}<extra></extra>",
        },
        {
            "type": "scatter",
            "mode": "markers",
            "x": [delay],
            "y": [tip],
            "marker": {"size": 9, "color": COLORS["resist"]},
            "showlegend": False,
            "hoverinfo": "skip",
        },
        _value_label(
            delay, tip, f"{tip:.2f}", color=COLORS["resist"], position="top right"
        ),
        {
            "type": "scatter",
            "mode": "markers",
            "x": [decision_t],
            "y": [decision_val],
            "marker": {"size": 8, "color": COLORS["resist"]},
            "showlegend": False,
            "hoverinfo": "skip",
        },
        _value_label(
            decision_label_x,
            decision_val,
            f"{decision_val:.2f}",
            color=COLORS["resist"],
            position="middle left",
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
            "line": {"color": COLORS["resist"], "width": 1, "dash": "dot"},
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


def _ss_point_traces(agent, ss_events):
    traces = []
    y_max = 0
    for i, (delay, reward) in enumerate(ss_events):
        tip = agent.value_function(reward, 0)
        y_max = max(y_max, tip)
        traces.append(
            {
                "type": "scatter",
                "mode": "markers",
                "x": [delay],
                "y": [tip],
                "marker": {"size": 10, "color": COLORS["snack"]},
                "name": "snack",
                "showlegend": i == 0,
                "hovertemplate": "SS @ %{x}h<br>V=%{y:.3f}<extra></extra>",
            }
        )
        traces.append(
            _value_label(
                delay, tip, f"{tip:.2f}", color=COLORS["snack"], position="top right"
            )
        )
    return traces, y_max


def _aggregate_traces(action_values):
    snack_agg = action_values["snack"]
    resist_agg = action_values["resist"]
    # Further left than discounted decision-node labels
    label_x = -LABEL_AGG_OFFSET
    traces = [
        {
            "type": "scatter",
            "mode": "markers",
            "x": [0],
            "y": [snack_agg],
            "marker": {
                "size": 12,
                "color": COLORS["snack_agg"],
                "line": {"color": COLORS["snack"], "width": 1.2},
            },
            "name": "snack aggregate",
            "hovertemplate": "snack agg=%{y:.3f}<extra></extra>",
        },
        _value_label(
            label_x,
            snack_agg,
            f"{snack_agg:.2f}",
            color=COLORS["snack_agg"],
            position="middle left",
        ),
        {
            "type": "scatter",
            "mode": "markers",
            "x": [0],
            "y": [resist_agg],
            "marker": {
                "size": 12,
                "color": COLORS["resist_agg"],
                "line": {"color": COLORS["resist"], "width": 1.2},
            },
            "name": "resist aggregate",
            "hovertemplate": "resist agg=%{y:.3f}<extra></extra>",
        },
        _value_label(
            label_x,
            resist_agg,
            f"{resist_agg:.2f}",
            color=COLORS["resist_agg"],
            position="middle left",
        ),
    ]
    return traces, max(snack_agg, resist_agg)


def _day_xticks(x_max):
    ticks = list(range(0, x_max + 1, 12))
    labels = [str(t // 24 + 1) if t % 24 == 0 else "" for t in ticks]
    return ticks, labels


def sails_frame(agent, environment, state, *, day=None):
    """
    Same geometry as visualise_sails / visualizing_main.draw_frame.
    If day is set, title includes the day number (steppable mode).
    """
    ss_events, ll_events = foreseeable_events(agent, environment, state)
    n = len(ll_events)
    action, action_values = agent.choose_action(state, environment)

    traces = []
    shapes = []
    y_max = 0

    for i, (delay, reward) in enumerate(ll_events):
        decision_t = ss_events[i][0]
        sail_traces, sail_shapes, peak = _ll_sail_trace(
            agent,
            delay,
            reward,
            decision_t,
            label="resist" if i == 0 else None,
            show_legend=(i == 0),
        )
        traces.extend(sail_traces)
        shapes.extend(sail_shapes)
        y_max = max(y_max, peak)

    ss_traces, ss_ymax = _ss_point_traces(agent, ss_events)
    traces.extend(ss_traces)
    y_max = max(y_max, ss_ymax)

    agg_traces, agg_ymax = _aggregate_traces(action_values)
    traces.extend(agg_traces)
    y_max = max(y_max, agg_ymax)

    x_max = max(
        max((d for d, _ in ll_events), default=1),
        max((d for d, _ in ss_events), default=1),
    )
    # Extra left room for aggregate labels (furthest left numerals)
    pad_x = max(LABEL_AGG_OFFSET + 6, x_max * 0.12)
    xticks, xticklabels = _day_xticks(x_max)

    if day is not None:
        title = (
            f"Day {day}  |  Choice sails with a horizon of {n} days  |  "
            f"Agent would choose {action}"
        )
    else:
        title = (
            f"Choice sails with a horizon of {n} days  |  "
            f"Agent would choose {action}"
        )

    return {
        "kind": "sails",
        "traces": traces,
        "layout": {
            "title": title,
            "xaxis": {
                "title": "Day",
                "range": [-pad_x, x_max + pad_x],
                "tickvals": xticks,
                "ticktext": xticklabels,
            },
            "yaxis": {
                "title": "Perceived value",
                "range": [0, y_max * 1.25 if y_max else 1],
            },
            "shapes": shapes,
            "legend": {"orientation": "h", "y": -0.18},
            "margin": {"l": 60, "r": 20, "t": 50, "b": 80},
            "height": 520,
        },
        "meta": {
            "day": day,
            "horizon": n,
            "choice": action,
            "action_values": action_values,
            "ll_ss_gap": action_values["resist"] - action_values["snack"],
        },
    }


def aggregate_frame(agent, environment, state):
    """Bundled resist sail + snack aggregate, matching aggregate_sail.py."""
    action, action_values = agent.choose_action(state, environment)
    ss_events = agent.choice_horizon(state, "snack", environment)
    ll_events = agent.choice_horizon(state, "resist", environment)
    ll_delay = environment.rewards["LL"]["delay"]

    def bundle_value_at(events, at_time):
        return sum(
            agent.value_function(reward, delay - at_time)
            for delay, reward in events
            if delay >= at_time
        )

    xs = list(range(ll_delay + 1))
    ys = [bundle_value_at(ll_events, t) for t in xs]
    ss_value = bundle_value_at(ss_events, 0)
    perceived_ll = ys[0] if ys else 0
    ll_tip = ys[-1] if ys else 0
    y_max = max(max(ys) if ys else 0, ss_value)

    traces = [
        {
            "type": "scatter",
            "mode": "lines",
            "x": xs,
            "y": ys,
            "line": {"color": COLORS["resist"], "width": 2.5},
            "name": "resist",
            "hovertemplate": "t=%{x}h<br>V=%{y:.3f}<extra></extra>",
        },
        {
            "type": "scatter",
            "mode": "markers+text",
            "x": [0],
            "y": [perceived_ll],
            "text": [f"{perceived_ll:.2f}"],
            "textposition": "top left",
            "marker": {"size": 10, "color": COLORS["resist"]},
            "showlegend": False,
            "hoverinfo": "skip",
        },
        {
            "type": "scatter",
            "mode": "markers+text",
            "x": [ll_delay],
            "y": [ll_tip],
            "text": [f"{ll_tip:.2f}"],
            "textposition": "top right",
            "marker": {"size": 10, "color": COLORS["resist"]},
            "showlegend": False,
            "hoverinfo": "skip",
        },
        {
            "type": "scatter",
            "mode": "markers+text",
            "x": [0],
            "y": [ss_value],
            "text": [f"{ss_value:.2f}"],
            "textposition": "bottom left",
            "marker": {"size": 10, "color": COLORS["snack"]},
            "name": "snack",
            "hovertemplate": "snack=%{y:.3f}<extra></extra>",
        },
    ]
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
            "title": f"Bundled value on day 1 |  Agent would choose {action}",
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
            "height": 520,
        },
        "meta": {
            "choice": action,
            "action_values": action_values,
        },
    }


def hyperbolic_frame(reward, delay, discount_factor):
    """
    Static full sail at remaining_delay = delay (hyperbolic_curve.py maths,
    using global LL reward/delay and agent discount_factor).
    """
    delay = int(delay)
    xs = list(range(delay + 1))
    ys = [reward / (1 + discount_factor * (delay - t)) for t in xs]
    perceived = reward / (1 + discount_factor * delay) if delay >= 0 else reward
    y_max = reward + 1

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
            "marker": {"size": 11, "color": "black"},
            "name": "perceived at t=0",
            "hovertemplate": "perceived=%{y:.3f}<extra></extra>",
        },
    ]
    shapes = [
        {
            "type": "line",
            "x0": delay,
            "x1": delay,
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
            "title": f"Hyperbolic sail | reward={reward}, delay={delay}h, k={discount_factor}",
            "xaxis": {
                "title": "Time steps from now",
                "range": [-0.2, max(delay, 1)],
            },
            "yaxis": {
                "title": "Perceived value",
                "range": [0, y_max],
            },
            "shapes": shapes,
            "legend": {"orientation": "h", "y": -0.15},
            "margin": {"l": 60, "r": 20, "t": 50, "b": 70},
            "height": 520,
        },
        "meta": {
            "reward": reward,
            "delay": delay,
            "discount_factor": discount_factor,
            "perceived": perceived,
        },
    }


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
    if reward_LL is None:
        reward_LL = environment.rewards["LL"]["value"]

    delays = delay_list(delay_max, delay_step)
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

    delay_lo = min(delays) if delays else 0
    delay_hi = max(delays) if delays else 1
    traces = [
        {
            "type": "scatter",
            "mode": "lines",
            "x": xs,
            "y": c["max_ss"],
            "line": {
                "width": 2.5,
                "color": sticky_delay_rgba(c["delay_LL"], delay_lo, delay_hi),
            },
            "name": f"LL delay = {c['delay_LL']}h",
            "hovertemplate": "s=%{x:.2f}<br>max SS=%{y:.3f}<extra></extra>",
        }
        for c in curves
    ]

    return {
        "kind": "sticky",
        "traces": traces,
        "layout": {
            "title": (
                f"Max SS resisted | LL={reward_LL}, horizon={horizon}, "
                f"discount_factor={agent.discount_factor}"
            ),
            "xaxis": {
                "title": "Stickiness s",
                "range": [0, 1],
            },
            "yaxis": {
                "title": "Max SS resisted",
                "range": [0, ymax],
            },
            "legend": {"orientation": "h", "y": -0.22},
            "margin": {"l": 60, "r": 20, "t": 50, "b": 80},
            "height": 520,
        },
        "meta": {
            "stickiness": xs,
            "delays": delays,
            "reward_LL": reward_LL,
            "horizon": horizon,
            "discount_factor": agent.discount_factor,
        },
    }
