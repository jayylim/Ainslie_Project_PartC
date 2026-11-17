'''
Same as sticky_sails (sails portion only), except the user can step through the environment.
Accumulated rewards are not tallied or considered yet (stateless agent).

At each decision the figure is redrawn from the current state; click / keypress
advances to the next day via environment.step.

At foresight day t (hours_from_now = t * decision_interval):

  stick_p_t = 1/2 * (1 + (2s - 1)^t)
  EV_t(a)   = stick_p_t * V_hyp(a, t) + (1 - stick_p_t) * V_hyp(b, t)
  V(a)      = sum_t EV_t(a)

When s = 1, EV_t collapses to pure commitment and V(a) matches HypAgent.

Run from the package root:

    python -m visual_demos.sticky_step
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
from main import agent_params, env_params, max_steps, stickiness

COLORS = {
    "snack": "#c44e52",
    "resist": "#4c72b0",
    # Sail bases; later foresight days fade toward white (labels stay opaque)
    "snack_sail": "#c44e52",
    "resist_sail": "#4c72b0",
}


def fade(base, i, n, max_fade=0.72):
    """i=0 full base colour; i=n-1 faded toward white. Scales with horizon n."""
    rgb = matplotlib.colors.to_rgb(base)
    t = (i / (n - 1) if n > 1 else 0.0) * max_fade
    return tuple(c + (1.0 - c) * t for c in rgb)


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


def plot_ll_sail(axes, agent, delay, reward, decision_t, color, *, label_base, label=None):
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
    axes.text(delay, tip, f"  {tip:.2f}", color=label_base, va="bottom", ha="left", size=11)

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
        color=label_base,
        va="bottom",
        ha=ha,
        size=11,
    )

    return max(ys) if ys else 0


def plot_ss_sails(axes, agent, ss_events, base_color):
    """SS sails feed back to t=0.
    Colour fades toward white as foresight index increases (fading scales with horizon).
    Data labels stay fully opaque base colour.
    """
    y_max = 0
    n = len(ss_events)
    for i, (delay, reward) in enumerate(ss_events):
        color = fade(base_color, i, n)
        label = "snack" if i == 0 else None
        if delay == 0:
            tip = agent.value_function(reward, 0)
            y_max = max(y_max, tip)
            axes.scatter(delay, tip, s=55, color=color, zorder=6, label=label)
            axes.text(
                delay,
                tip,
                f"  {tip:.2f}",
                color=base_color,
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
            color=base_color,
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


def draw_frame(axes, agent, environment, state, steps):
    """Redraw the sticky sails view from the current decision state."""
    ss_events, ll_events = outcome_events(agent, environment, state)
    n = agent.get_horizon(state, environment)

    # obtain current choice
    action, action_values = agent.choose_action(state, environment)

    axes.clear()
    y_max = 0

    n_sails = len(ll_events)
    for i, (delay, reward) in enumerate(ll_events):
        # Decision node for this foresight day (SS offer time = t * interval)
        decision_t = ss_events[i][0]
        sail_color = fade(COLORS["resist_sail"], i, n_sails)
        peak = plot_ll_sail(
            axes,
            agent,
            delay,
            reward,
            decision_t,
            sail_color,
            label_base=COLORS["resist_sail"],
            label="resist" if i == 0 else None,
        )
        y_max = max(y_max, peak)

    # ensure y_max is actually the highest point of the graph
    y_max = max(y_max, plot_ss_sails(axes, agent, ss_events, COLORS["snack_sail"]))
    y_max = max(y_max, plot_aggregates(axes, action_values, action))

    # Fixed x-domain: agent horizon in hours (does not shrink as remaining days fall)
    interval = environment.decision_interval
    horizon_days = agent.horizon if agent.horizon is not None else n
    x_max = horizon_days * interval
    pad_x = max(8, x_max * 0.12)
    axes.set_xlim(-pad_x, x_max + pad_x)
    axes.set_ylim(0, y_max * 1.25 if y_max else 1)

    xticks = list(range(0, x_max + 1, 12))
    axes.set_xticks(xticks)
    axes.set_xticklabels([str(t) for t in xticks])
    axes.set_xlabel("Hours from now")
    axes.set_ylabel("Perceived value")
    axes.set_title(
        f"Day {steps}  |  Sticky sails s={stickiness} | horizon={n} days  |  "
        f"Agent would choose {action}"
    )
    axes.legend(loc="upper right")

    snack_days = agent.choice_horizon(state, "snack", environment)
    resist_days = agent.choice_horizon(state, "resist", environment)

    # show tables of discounted values and compounded stickiness
    print(f"Day {steps}  |  Horizon = {n}  stickiness s = {stickiness}")
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

    # show tables of expected day values
    print(
        "Table 2 — Sticky expected day values\n"
        "  EV_t | choice first = stick_p * V_hyp(a) + (1-stick_p) * V_hyp(b)"
    )
    print(f"{'hours':>6}  {'EV snack-first':>16}  {'EV resist-first':>16}")
    for (t, ev_s), (_, ev_r) in zip(snack_days, resist_days):
        hours = t * interval
        print(f"{hours:6d}  {ev_s:16.4f}  {ev_r:16.4f}")
    print(f"{'sum':>6}  {action_values['snack']:16.6f}  {action_values['resist']:16.6f}")
    print()
    print(f"Would choose {action}")

    # Sanity: sticky LL − SS aggregate at the current decision
    ll_ss_gap = action_values["resist"] - action_values["snack"]
    print(f"LL−SS (sticky) at day 0: {ll_ss_gap:.4f}")
    print()

    return action, ll_ss_gap


def main():
    environment = TimelineEnvironment(**env_params)
    agent = StickyHypAgent(**agent_params, stickiness=stickiness)
    state = environment.reset()

    finished = False
    steps = 0
    ll_ss_gaps = []  # sticky LL−SS at day 0 each step

    fig, axes = plt.subplots(figsize=(10, 5))
    print("Starting sticky visualisation (click / keypress to advance)...")

    while not finished and (max_steps is None or steps < max_steps):
        steps += 1
        action, ll_ss_gap = draw_frame(axes, agent, environment, state, steps)
        ll_ss_gaps.append(ll_ss_gap)
        state, finished = environment.step(action)
        plt.tight_layout()
        plt.pause(0.05)
        plt.waitforbuttonpress()

    print(f"Day {steps} LL−SS gaps: {ll_ss_gaps}")
    print("Done.")
    plt.show()


if __name__ == "__main__":
    main()
