'''
Sticky choice sails: same SS / LL sail geometry as visualise_sails, but
aggregates use stickiness (StickyHypAgent / sticky_decisions).

At foresight day t (hours_from_now = t * decision_interval):

  stick_p_t = 1/2 * (1 + (2s - 1)^t)
  EV_t(a)   = stick_p_t * V_hyp(a, t) + (1 - stick_p_t) * V_hyp(b, t)
  V(a)      = sum_t EV_t(a)

Light sails = pure hyperbolic SS / LL trajectories (day_value building blocks).
Opaque scatters at t=0 = sticky aggregates V(a) (StickyHypAgent.evaluate_action).
Console prints V_hyp and sticky EV as two separate tables (hours from now).

When s = 1, EV_t collapses to pure commitment and V(a) matches HypAgent.

Run from the package root:

    python -m visual_demos.visualise_sticky_sails
'''

import matplotlib
import matplotlib.pyplot as plt

import sys
from pathlib import Path

# Debugger / direct run put visual_demos/ on sys.path; package root is one up
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    matplotlib.use("MacOSX")
except Exception:
    pass

from agents.sticky import StickyHypAgent
from environments.timeline import TimelineEnvironment
from main import agent_params, env_params, stickiness

# --- Optional sticky foresight panels (easy to reject) -------------------
# Set False, or delete this flag + the call in main(), and delete
# visual_demos/sticky_foresight_panels.py
SHOW_STICKY_BREAKDOWN = True
if SHOW_STICKY_BREAKDOWN:
    from visual_demos.sticky_foresight_panels import show_sticky_breakdown
# ------------------------------------------------------------------------

COLORS = {
    "snack": "#c44e52",
    "resist": "#4c72b0",
    # Lighter tint for option sails; aggregates use the opaque snack / resist colours
    "snack_sail": "#e89a9e",
    "resist_sail": "#8fb0d9",
}


def outcome_events(agent, environment, state):
    """(delay, reward) SS / LL pairs at each foresight day.

    Same delay structure as StickyHypAgent.day_value / HypAgent.choice_horizon:
    outcomes are shifted by t * decision_interval from the current decision.
    """
    horizon = agent.get_horizon(state, environment)
    interval = environment.decision_interval
    ss_events = []
    ll_events = []
    for t in range(horizon):
        _, ss_outcomes = environment.simulate(state, "snack")
        _, ll_outcomes = environment.simulate(state, "resist")
        for reward, delay in ss_outcomes.values():
            ss_events.append((delay + t * interval, reward))
        for reward, delay in ll_outcomes.values():
            ll_events.append((delay + t * interval, reward))
    return ss_events, ll_events


def plot_ll_sail(axes, agent, delay, reward, decision_t, color, *, label=None):
    """Hyperbolic sail for one LL: full curve from t=0 to delivery."""
    xs, ys = [], []
    for t in range(0, delay + 1):
        xs.append(t)
        ys.append(agent.value_function(reward, delay - t))

    axes.plot(xs, ys, color=color, label=label)
    axes.axvline(delay, linestyle=":", color=color, alpha=0.45)
    axes.axvline(decision_t, linestyle=":", color="black", alpha=0.55)

    tip = ys[-1]
    axes.scatter(delay, tip, s=50, color=color, zorder=5)
    axes.text(delay, tip, f"  {tip:.2f}", color=color, va="bottom", ha="left", size=11)

    # Pure V_hyp at this sail's decision node (remaining delay from that choice)
    decision_val = agent.value_function(reward, delay - decision_t)
    axes.scatter(decision_t, decision_val, s=40, color=color, zorder=5)
    ha = "right" if decision_t == 0 else "left"
    prefix = "" if ha == "right" else "  "
    offset = -1 if ha == "right" else 0
    axes.text(
        decision_t + offset,
        decision_val,
        f"{prefix}{decision_val:.2f}",
        color=color,
        va="bottom",
        ha=ha,
        size=11,
    )

    return max(ys) if ys else 0


def plot_ss_sails(axes, agent, ss_events, color):
    """SS at the current decision is a tip scatter; later SS get sails back to t=0."""
    y_max = 0
    for i, (delay, reward) in enumerate(ss_events):
        label = "snack" if i == 0 else None
        if delay == 0:
            tip = agent.value_function(reward, 0)
            y_max = max(y_max, tip)
            axes.scatter(delay, tip, s=55, color=color, zorder=6, label=label)
            axes.text(
                delay,
                tip,
                f"  {tip:.2f}",
                color=color,
                va="bottom",
                ha="left",
                size=11,
            )
            continue

        xs, ys = [], []
        for t in range(0, delay + 1):
            xs.append(t)
            ys.append(agent.value_function(reward, delay - t))

        axes.plot(xs, ys, color=color, label=label)
        tip = ys[-1]
        y_max = max(y_max, max(ys) if ys else 0)
        axes.scatter(delay, tip, s=55, color=color, zorder=6)
        axes.text(
            delay,
            tip,
            f"  {tip:.2f}",
            color=color,
            va="bottom",
            ha="left",
            size=11,
        )
    return y_max


def plot_aggregates(axes, action_values, chosen):
    """Sticky bundled snack / resist values at the current decision (t=0)."""
    snack_agg = action_values["snack"]
    resist_agg = action_values["resist"]

    axes.scatter(
        0,
        snack_agg,
        s=70,
        color=COLORS["snack"],
        zorder=7,
        edgecolors=COLORS["snack"],
        linewidths=1.2,
        label="snack aggregate",
    )
    axes.text(
        -2,
        snack_agg + 1,
        f"{snack_agg:.2f}",
        color=COLORS["snack"],
        va="bottom",
        ha="right",
        size=11,
        fontweight="bold" if chosen == "snack" else "normal",
    )

    axes.scatter(
        0,
        resist_agg,
        s=70,
        color=COLORS["resist"],
        zorder=7,
        edgecolors=COLORS["resist"],
        linewidths=1.2,
        label="resist aggregate",
    )
    axes.text(
        -2,
        resist_agg,
        f"{resist_agg:.2f}",
        color=COLORS["resist"],
        va="bottom",
        ha="right",
        size=11,
        fontweight="bold" if chosen == "resist" else "normal",
    )

    return max(snack_agg, resist_agg)


def main():
    environment = TimelineEnvironment(**env_params)
    agent = StickyHypAgent(**agent_params, stickiness=stickiness)
    state = environment.reset()

    ss_events, ll_events = outcome_events(agent, environment, state)
    n = agent.get_horizon(state, environment)

    # Sticky aggregates + greedy choice (same as StickyHypAgent / sticky_decisions)
    action, action_values = agent.choose_action(state, environment)

    fig, axes = plt.subplots(figsize=(10, 5))
    y_max = 0

    for i, (delay, reward) in enumerate(ll_events):
        # Decision node for this foresight day (SS offer time = t * interval)
        decision_t = ss_events[i][0]
        peak = plot_ll_sail(
            axes,
            agent,
            delay,
            reward,
            decision_t,
            COLORS["resist_sail"],
            label="resist" if i == 0 else None,
        )
        y_max = max(y_max, peak)

    y_max = max(y_max, plot_ss_sails(axes, agent, ss_events, COLORS["snack_sail"]))
    y_max = max(y_max, plot_aggregates(axes, action_values, action))

    x_max = max(
        max((d for d, _ in ll_events), default=1),
        max((d for d, _ in ss_events), default=1),
    )
    pad_x = max(8, x_max * 0.12)
    axes.set_xlim(-pad_x, x_max + pad_x)
    axes.set_ylim(0, y_max * 1.25 if y_max else 1)

    xticks = list(range(0, x_max + 1, 12))
    axes.set_xticks(xticks)
    axes.set_xticklabels([str(t) for t in xticks])
    axes.set_xlabel("Hours from now (decision node)")
    axes.set_ylabel("Perceived value")
    axes.set_title(
        f"Sticky sails | s={stickiness} | horizon={n} days  |  "
        f"Agent would choose {action}"
    )
    axes.legend(loc="upper right")

    interval = environment.decision_interval
    snack_days = agent.choice_horizon(state, "snack", environment)
    resist_days = agent.choice_horizon(state, "resist", environment)

    print(f"Horizon = {n}  stickiness s = {stickiness}")
    print()
    print(
        "Table 1 — Pure hyperbolic day values\n"
        "  V_hyp(choice) from now at each decision node"
    )
    print(f"{'hours':>6}  {'stick_p':>10}  {'V_hyp(snack)':>14}  {'V_hyp(resist)':>14}")
    for (t, _), (_, _) in zip(snack_days, resist_days):
        hours = t * interval
        v_s = agent.day_value(state, "snack", environment, t)
        v_r = agent.day_value(state, "resist", environment, t)
        print(
            f"{hours:6d}  {agent.stick_p_t(t):10.4f}  {v_s:14.4f}  {v_r:14.4f}"
        )
    print()
    print(
        "Table 2 — Sticky expected day values\n"
        "  EV_t | choice first = stick_p * V_hyp(a) + (1-stick_p) * V_hyp(b)"
    )
    print(f"{'hours':>6}  {'EV snack-first':>16}  {'EV resist-first':>16}")
    for (t, ev_s), (_, ev_r) in zip(snack_days, resist_days):
        hours = t * interval
        print(f"{hours:6d}  {ev_s:16.4f}  {ev_r:16.4f}")
    print(
        f"{'sum':>6}  {action_values['snack']:16.6f}  {action_values['resist']:16.6f}"
    )
    print()
    print(f"Would choose {action}")

    plt.tight_layout()

    # Optional second figure — see SHOW_STICKY_BREAKDOWN / sticky_foresight_panels.py
    if SHOW_STICKY_BREAKDOWN:
        show_sticky_breakdown(
            agent, environment, state, action_values, action, stickiness
        )

    plt.show()


if __name__ == "__main__":
    main()
