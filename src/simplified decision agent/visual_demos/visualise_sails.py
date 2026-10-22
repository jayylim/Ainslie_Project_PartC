'''
One sail per foreseeable LL, one scatter per foreseeable SS.

Each reward is discounted from the current decision, then
those values are summed (see agents.hyperbolic.HypAgent.aggregate_value).

At day 1 the agent plans {horizon} steps ahead. For each future SS / LL:

  1. Discount that single reward with its delay from now
  2. LL as a sail y(t) = V(reward, delay - t) from t=0; SS as a scatter at the tip

Black dotted lines mark decision nodes (when that LL's SS is offered).
Lighter scatters at t=0 show the bundled snack / resist aggregates
(same as HypAgent.evaluate_action / aggregate_value).

Run from the package root:

    python -m visual_demos.visualise_sails
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

from agents.hyperbolic import HypAgent
from environments.timeline import TimelineEnvironment
from main import agent_params, env_params

COLORS = {
    "snack": "#c44e52",
    "resist": "#4c72b0",
    "snack_agg": "#e89a9e", #aggregate colours are lighter versions of the individual colours
    "resist_agg": "#8fb0d9",
}


def foreseeable_events(agent, environment, state):
    # 1 or horizon as generated in agent
    return (
        agent.choice_horizon(state, "snack", environment),
        agent.choice_horizon(state, "resist", environment),
    )


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

    # Value at this sail's decision node (remaining delay from that choice)

      ### maybe change to t=0 value (but looks ugly on graph cus too overcrowded) 

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


def plot_ss_points(axes, agent, ss_events, color):
    """One scatter per foreseeable SS tip (full reward at delivery)."""
    y_max = 0
    for i, (delay, reward) in enumerate(ss_events):
        tip = agent.value_function(reward, 0)
        y_max = max(y_max, tip)
        label = "snack" if i == 0 else None
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
    return y_max


def plot_aggregates(axes, action_values):
    """Bundled snack / resist values at the current decision (t=0)."""
    snack_agg = action_values["snack"]
    resist_agg = action_values["resist"]

    axes.scatter(
        0,
        snack_agg,
        s=70,
        color=COLORS["snack_agg"],
        zorder=7,
        edgecolors=COLORS["snack"],
        linewidths=1.2,
        label="snack aggregate",
    )
    axes.text(
        -2,
        snack_agg+1,
        f"{snack_agg:.2f}",
        color=COLORS["snack_agg"],
        va="bottom",
        ha="right",
        size=11,
    )

    axes.scatter(
        0,
        resist_agg,
        s=70,
        color=COLORS["resist_agg"],
        zorder=7,
        edgecolors=COLORS["resist"],
        linewidths=1.2,
        label="resist aggregate",
    )
    axes.text(
        -2,
        resist_agg,
        f"{resist_agg:.2f}",
        color=COLORS["resist_agg"],
        va="bottom",
        ha="right",
        size=11,
    )

    return max(snack_agg, resist_agg)


def main():
    environment = TimelineEnvironment(**env_params)
    agent = HypAgent(**agent_params)
    state = environment.reset()

    ss_events, ll_events = foreseeable_events(agent, environment, state)
    n = len(ll_events) #no. of LL sails == no. of decisions to be made, for plot labels

    # obtain current choice
    action, action_values = agent.choose_action(state, environment)

    fig, axes = plt.subplots(figsize=(10, 5))
    y_max = 0

    for i, (delay, reward) in enumerate(ll_events):
        # SS offer / decision node for this step (same index in the plan)
        decision_t = ss_events[i][0]
        peak = plot_ll_sail(
            axes,
            agent,
            delay,
            reward,
            decision_t,
            COLORS["resist"],
            label="resist" if i == 0 else None,
        )
        y_max = max(y_max, peak)

    # ensure y_max is actually the highest point of the graph
    y_max = max(y_max, plot_ss_points(axes, agent, ss_events, COLORS["snack"]))
    y_max = max(y_max, plot_aggregates(axes, action_values))

    # ensure x_max is actually the longest delay; its normally from LL-events anyway
    x_max = max(
        max((d for d, _ in ll_events), default=1),
        max((d for d, _ in ss_events), default=1),
    )
    # Extra padding so left labels, tip labels, and sail peaks are not clipped
    pad_x = max(8, x_max * 0.12)
    axes.set_xlim(-pad_x, x_max + pad_x)
    axes.set_ylim(0, y_max * 1.25 if y_max else 1)

    xticks = list(range(0, x_max + 1, 12))
    axes.set_xticks(xticks)
    axes.set_xticklabels([str(t // 24 + 1) if t % 24 == 0 else "" for t in xticks])
    axes.set_xlabel("Day")
    axes.set_ylabel("Perceived value")
    axes.set_title(f"Choice sails with a horizon of {n} days  |  Agent would choose {action}")
    axes.legend(loc="upper right")

    # Day-1 perceived value of each outcome (same terms as aggregate_value)
    print(f"Horizon = {n}")
    print(f"SS delay | V(reward, delay) from now:")
    for delay, reward in ss_events:
        v = agent.value_function(reward, delay)
        print(f"Day {delay//24 + 1} | {delay}h  Value={v:.4f}")
    print(f"LL delay | V(reward, delay) from now:")
    for delay, reward in ll_events:
        v = agent.value_function(reward, delay)
        print(f"Day {delay//12}   {delay}h  Value={v:.4f}")
    print(
        f"Current choice values: snack={action_values['snack']:.4f}  "
        f"resist={action_values['resist']:.4f}"
    )
    print(f"Would choose {action}")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
