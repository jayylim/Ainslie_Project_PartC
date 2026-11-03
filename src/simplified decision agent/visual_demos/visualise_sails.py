'''
One sail per foreseeable LL, one sail (or tip scatter) per foreseeable SS.

Each reward is discounted from the current decision, then
those values are summed (see agents.hyperbolic.HypAgent.aggregate_value).

At day 1 the agent plans {horizon} steps ahead. For each future SS / LL:

  1. Discount that single reward with its delay from now
  2. Sail y(t) = V(reward, delay - t) from t=0 to delivery;
     the current SS (delay 0) is tip-only

Black dotted lines mark decision nodes (when that LL's SS is offered).
Opaque scatters at t=0 show the bundled snack / resist aggregates
(same as HypAgent.evaluate_action / aggregate_value). Later foresight sails
fade toward white; data labels stay fully opaque.

A side panel stacks each day's V_hyp contribution so the opaque aggregate
dots are visibly V(a) = sum_t V_hyp(a, t) under pure commitment.

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
    # Sail bases; later foresight days fade toward white (labels stay opaque)
    "snack_sail": "#c44e52",
    "resist_sail": "#4c72b0",
}


def fade(base, i, n, max_fade=0.72):
    """i=0 full base colour; i=n-1 faded toward white. Scales with horizon n."""
    rgb = matplotlib.colors.to_rgb(base)
    t = (i / (n - 1) if n > 1 else 0.0) * max_fade
    return tuple(c + (1.0 - c) * t for c in rgb)


def foreseeable_events(agent, environment, state):
    # 1 or horizon as generated in agent
    return (
        agent.choice_horizon(state, "snack", environment),
        agent.choice_horizon(state, "resist", environment),
    )


def plot_ll_sail(
    axes,
    agent,
    delay,
    reward,
    decision_t,
    color,
    *,
    label_base,
    label=None,
    show_labels=True,
):
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
    if show_labels:
        axes.text(delay, tip, f"  {tip:.2f}", color=label_base, va="bottom", ha="left", size=11)

    # Value at this sail's decision node (remaining delay from that choice)
    decision_val = agent.value_function(reward, delay - decision_t)
    axes.scatter(decision_t, decision_val, s=40, color=color, zorder=5)
    if show_labels:
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


def plot_ss_sails(axes, agent, ss_events, base_color, *, label_cutoff=None):
    """SS at the current decision is a tip scatter; later SS get sails back to t=0.

    Sail colour fades with foresight index; data labels stay fully opaque.
    Numeric tip labels only for indices before label_cutoff (None = all).
    """
    y_max = 0
    n = len(ss_events)
    if label_cutoff is None:
        label_cutoff = n
    for i, (delay, reward) in enumerate(ss_events):
        color = fade(base_color, i, n)
        label = "snack" if i == 0 else None
        show_labels = i < label_cutoff
        if delay == 0:
            tip = agent.value_function(reward, 0)
            y_max = max(y_max, tip)
            axes.scatter(delay, tip, s=55, color=color, zorder=6, label=label)
            if show_labels:
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
        if show_labels:
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
    """Bundled snack / resist values at the current decision (t=0)."""
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

#plotting some cumulative bar graph

def plot_aggregate_stack(axes, agent, ss_events, ll_events, action_values, chosen):
    """Σ bars only: each segment is that foresight day's V_hyp.

    Pure commitment (HypAgent): no stickiness mix. 
    """
    n = len(ll_events)
    if n == 0:
        return

    # One contribution per foresight day (delay, reward) → V from now
    snack_parts = [
        (delay, agent.value_function(reward, delay)) for delay, reward in ss_events
    ]
    resist_parts = [
        (delay, agent.value_function(reward, delay)) for delay, reward in ll_events
    ]
    snack_agg = action_values["snack"]
    resist_agg = action_values["resist"]

    snack_cmap = plt.cm.Reds
    resist_cmap = plt.cm.Blues

    def fade_cmap(cmap, i, n_seg):
        t = i / (n_seg - 1) if n_seg > 1 else 0.0
        return cmap(0.88 - 0.56 * t)

    x_snack, x_resist = 0.0, 1.0
    bar_w = 0.55
    bottom_s = 0.0
    bottom_r = 0.0
    min_label = 0.05 * max(reward for _, reward in ss_events)
    # (hours, height, segment mid-y) for side callouts
    snack_callouts = []
    resist_callouts = []

    for i in range(n):
        hours_s, h_s = snack_parts[i]
        hours_r, h_r = resist_parts[i]
        edge_s = 1.2 if chosen == "snack" else 0.5
        edge_r = 1.2 if chosen == "resist" else 0.5

        axes.bar(
            x_snack,
            h_s,
            width=bar_w,
            bottom=bottom_s,
            color=fade_cmap(snack_cmap, i, n),
            edgecolor="black",
            linewidth=edge_s,
        )
        axes.bar(
            x_resist,
            h_r,
            width=bar_w,
            bottom=bottom_r,
            color=fade_cmap(resist_cmap, i, n),
            edgecolor="black",
            linewidth=edge_r,
        )
        if h_s >= min_label:
            snack_callouts.append((hours_s, h_s, bottom_s + h_s / 2))
        if h_r >= min_label:
            resist_callouts.append((hours_r, h_r, bottom_r + h_r / 2))
        bottom_s += h_s
        bottom_r += h_r

    y_peak = max(snack_agg, resist_agg) or 1.0

    def spaced_ys(count, lo, hi):
        if count <= 0:
            return []
        if count == 1:
            return [(lo + hi) / 2]
        return [lo + (hi - lo) * i / (count - 1) for i in range(count)]

    # data point labels as callouts
    def draw_callouts(callouts, x_bar, x_text, ha):
        """Evenly spaced side labels with arrows into each stack segment."""
        ys = spaced_ys(len(callouts), y_peak * 0.06, y_peak * 0.94)
        x_edge = x_bar - bar_w / 2 if ha == "right" else x_bar + bar_w / 2
        for (hours, h, mid_y), y_lab in zip(callouts, ys):
            axes.annotate(
                f"{int(hours)}h\n{h:.2f}", # reward delay in hours
                xy=(x_edge, mid_y),
                xytext=(x_text, y_lab),
                ha=ha,
                va="center",
                fontsize=7,
                color="#4c72b0" if ha == "left" else "#c44e52", # right and left are substitute for resist and snack respectively
                arrowprops={
                    "arrowstyle": "->",
                    "color": "#666666",
                    "lw": 0.8,
                    "shrinkA": 0,
                    "shrinkB": 2,
                },
                zorder=8,
            )

    draw_callouts(snack_callouts, x_snack, x_snack - 0.85, "right")
    draw_callouts(resist_callouts, x_resist, x_resist + 0.85, "left")


    axes.text(
        x_snack - 0.75,
        y_peak * 1.04,
        f"Σ snack = {snack_agg:.2f}",
        ha="left",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        color=COLORS["snack"],
    )

    axes.text(
        x_resist + 0.75,
        y_peak * 1.04,
        f"Σ resist = {resist_agg:.2f}",
        ha="right",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        color=COLORS["resist"],
    )

    axes.set_xticks([x_snack, x_resist])
    axes.set_xticklabels(["snack", "resist"])
    axes.set_xlim(-1.25, 2.25)
    axes.set_ylim(0, y_peak * 1.18)
    axes.set_ylabel("Contribution from now")
    axes.set_title(
        r"Summation of discounted values"
        "\n"
        "(Aggregate option values)"
    )
    
    # axes.legend(loc="upper right", fontsize=7) # removed legend for now
    axes.grid(True, axis="y", alpha=0.25)


def main():
    environment = TimelineEnvironment(**env_params)
    agent = HypAgent(**agent_params)
    state = environment.reset()

    ss_events, ll_events = foreseeable_events(agent, environment, state)
    n = len(ll_events) #no. of LL sails == no. of decisions to be made, for plot labels

    # obtain current choice
    action, action_values = agent.choose_action(state, environment)

    fig, (ax_stack, axes) = plt.subplots(
        1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1, 3]}
    )

    plot_aggregate_stack(
        ax_stack, agent, ss_events, ll_events, action_values, action
    )

    y_max = 0

    n_sails = len(ll_events)
    # Numeric labels (decision nodes + peaks) only for the first half of foresight days
    label_cutoff = n_sails / 2
    for i, (delay, reward) in enumerate(ll_events):
        # SS offer / decision node for this step (same index in the plan)
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
            show_labels=i < label_cutoff,
        )
        y_max = max(y_max, peak)

    # ensure y_max is actually the highest point of the graph
    y_max = max(
        y_max,
        plot_ss_sails(
            axes, agent, ss_events, COLORS["snack_sail"], label_cutoff=label_cutoff
        ),
    )
    y_max = max(y_max, plot_aggregates(axes, action_values, action))

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
    axes.set_xticklabels([str(t) for t in xticks])
    axes.set_xlabel("Hours from now")
    axes.set_ylabel("Perceived value")
    axes.set_title(f"Choice sails with a horizon of {n} days  |  Agent would choose {action}")
    axes.legend(loc="upper right")


    # Day-1 perceived value of each outcome (same terms as aggregate_value)
    print(f"Horizon = {n}")
    print(f"Day| Hours from now , Perceived discounted value:")
    for delay, reward in ss_events:
        v = agent.value_function(reward, delay)
        print(f"Day {delay//24 + 1} | {delay}h  Value = {v:.4f}")
    print(f"Day| Hours from now , Perceived discounted value:")
    for delay, reward in ll_events:
        v = agent.value_function(reward, delay)
        print(f"Day {-(-delay//24)} | {delay}h  Value = {v:.4f}")
    print(
        f"Current choice values: snack={action_values['snack']:.4f}  "
        f"resist={action_values['resist']:.4f}"
    )
    print(f"Would choose {action}")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
