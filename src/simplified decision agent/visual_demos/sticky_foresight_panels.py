'''
Optional sticky foresight panels (companion to visualise_sticky_sails).

Easy to reject later:
  1. Delete this file, and
  2. Remove the SHOW_STICKY_BREAKDOWN block / import from visualise_sticky_sails.py

Figure layout (one window, two panels):

  Left  — tree-chain persistence
          stick_p_t = P(a_t = a | a_0 = a) = 1/2 * (1 + (2s - 1)^t)
          This is the weight multiplied onto today's-action V_hyp at day t;
          (1 - stick_p_t) weights the other action.

  Right — stacked EV contributions at each foresight hour
          For each candidate a at hours h = t * decision_interval:
            bottom segment = stick_p_t     * V_hyp(a, t)
            top segment    = (1-stick_p_t) * V_hyp(b, t)
            bar height     = EV_t(a)
          Two bars per hour (snack-first | resist-first).
          Side totals show V(a) = sum_t EV_t(a).
'''

import matplotlib.pyplot as plt
import numpy as np

COLORS = {
    "snack": "#c44e52",
    "resist": "#4c72b0",
    "snack_light": "#e89a9e",
    "resist_light": "#8fb0d9",
    "persist": "#2f4f4f",
    "switch": "#b8860b",
}


def foresight_rows(agent, environment, state):
    """Rows aligned with StickyHypAgent / sticky_decisions tables."""
    interval = environment.decision_interval
    horizon = agent.get_horizon(state, environment)
    rows = []
    for t in range(horizon):
        stick_p = agent.stick_p_t(t)
        v_snack = agent.day_value(state, "snack", environment, t)
        v_resist = agent.day_value(state, "resist", environment, t)
        # snack-first: same=snack, other=resist
        snack_same = stick_p * v_snack
        snack_other = (1.0 - stick_p) * v_resist
        # resist-first: same=resist, other=snack
        resist_same = stick_p * v_resist
        resist_other = (1.0 - stick_p) * v_snack
        rows.append(
            {
                "t": t,
                "hours": t * interval,
                "stick_p": stick_p,
                "v_snack": v_snack,
                "v_resist": v_resist,
                "snack_same": snack_same,
                "snack_other": snack_other,
                "ev_snack": snack_same + snack_other,
                "resist_same": resist_same,
                "resist_other": resist_other,
                "ev_resist": resist_same + resist_other,
            }
        )
    return rows


def plot_persistence_chain(axes, rows, s):
    """Left panel: foresight persistence (tree-chain probability)."""
    hours = [r["hours"] for r in rows]
    stick_p = [r["stick_p"] for r in rows]
    switch_p = [1.0 - p for p in stick_p]

    axes.plot(
        hours,
        stick_p,
        color=COLORS["persist"],
        marker="o",
        linewidth=2,
        label=r"$p_t=$ stick weight on today's action",
    )
    axes.plot(
        hours,
        switch_p,
        color=COLORS["switch"],
        marker="s",
        linestyle="--",
        linewidth=1.5,
        label=r"$1-p_t=$ weight on the other action",
    )
    axes.axhline(0.5, color="gray", linestyle=":", alpha=0.6, label="long-run 1/2")

    # Small chain marks: at each step after t=0, annotate local stickiness s
    if len(hours) >= 2:
        axes.annotate(
            f"step stickiness s={s:g}",
            xy=(hours[1], stick_p[1]),
            xytext=(hours[1] + max(hours[-1] * 0.05, 8), stick_p[1] + 0.08),
            fontsize=9,
            color=COLORS["persist"],
            arrowprops=dict(arrowstyle="->", color=COLORS["persist"], lw=0.8),
        )

    axes.set_ylim(-0.05, 1.08)
    axes.set_xlabel("Hours from now")
    axes.set_ylabel("Foresight probability weight")
    axes.set_title(
        r"Tree-chain persistence  $p_t=\frac{1}{2}(1+(2s-1)^t)$"
        "\n(multiplies that day's $V_{hyp}$ of today's action)"
    )
    axes.legend(loc="best", fontsize=8)
    axes.grid(True, alpha=0.25)


def plot_ev_stacked_bars(axes, rows, action_values, chosen):
    """Right panel: stacked hyp-weighted pieces → EV height at each hour."""
    hours = np.array([r["hours"] for r in rows], dtype=float)
    n = len(hours)
    if n == 0:
        return

    # Spacing: two bars per hour group
    group_gap = max(hours[1] - hours[0], 24) if n > 1 else 24
    bar_w = group_gap * 0.28
    offset = bar_w * 0.55

    snack_same = np.array([r["snack_same"] for r in rows])
    snack_other = np.array([r["snack_other"] for r in rows])
    resist_same = np.array([r["resist_same"] for r in rows])
    resist_other = np.array([r["resist_other"] for r in rows])

    x_snack = hours - offset
    x_resist = hours + offset

    # snack-first stack: bottom = p * V_snack, top = (1-p) * V_resist
    axes.bar(
        x_snack,
        snack_same,
        width=bar_w,
        color=COLORS["snack"],
        edgecolor="white",
        linewidth=0.4,
        label=r"snack-first: $p_t\,V_{hyp}(snack)$",
    )
    axes.bar(
        x_snack,
        snack_other,
        width=bar_w,
        bottom=snack_same,
        color=COLORS["resist_light"],
        edgecolor="white",
        linewidth=0.4,
        label=r"snack-first: $(1-p_t)\,V_{hyp}(resist)$",
    )

    # resist-first stack: bottom = p * V_resist, top = (1-p) * V_snack
    axes.bar(
        x_resist,
        resist_same,
        width=bar_w,
        color=COLORS["resist"],
        edgecolor="white",
        linewidth=0.4,
        label=r"resist-first: $p_t\,V_{hyp}(resist)$",
    )
    axes.bar(
        x_resist,
        resist_other,
        width=bar_w,
        bottom=resist_same,
        color=COLORS["snack_light"],
        edgecolor="white",
        linewidth=0.4,
        label=r"resist-first: $(1-p_t)\,V_{hyp}(snack)$",
    )

    # Aggregate totals as reference bars to the right of the last group
    total_x0 = hours[-1] + group_gap * 1.35
    v_snack = action_values["snack"]
    v_resist = action_values["resist"]
    axes.bar(
        total_x0 - offset,
        v_snack,
        width=bar_w,
        color=COLORS["snack"],
        alpha=0.85,
        edgecolor="black",
        linewidth=1.2 if chosen == "snack" else 0.6,
        label=r"aggregate $V(snack)=\sum EV_t$",
    )
    axes.bar(
        total_x0 + offset,
        v_resist,
        width=bar_w,
        color=COLORS["resist"],
        alpha=0.85,
        edgecolor="black",
        linewidth=1.2 if chosen == "resist" else 0.6,
        label=r"aggregate $V(resist)=\sum EV_t$",
    )
    axes.text(
        total_x0,
        max(v_snack, v_resist) * 1.04,
        "sums → today",
        ha="center",
        va="bottom",
        fontsize=9,
    )

    axes.set_xticks(list(hours) + [total_x0])
    axes.set_xticklabels([str(int(h)) for h in hours] + ["Σ"])
    axes.set_xlabel("Hours from now")
    axes.set_ylabel("Value contribution (from now)")
    axes.set_title(
        "Stacked day contributions\n"
        r"bar height $=EV_t$; opaque totals $=V(a)=\sum_t EV_t$"
    )
    axes.legend(loc="upper right", fontsize=7)
    axes.grid(True, axis="y", alpha=0.25)


def show_sticky_breakdown(agent, environment, state, action_values, chosen, s):
    """Open the optional two-panel sticky foresight figure."""
    rows = foresight_rows(agent, environment, state)
    fig, (ax_p, ax_ev) = plt.subplots(
        1, 2, figsize=(13, 5), constrained_layout=True
    )
    plot_persistence_chain(ax_p, rows, s)
    plot_ev_stacked_bars(ax_ev, rows, action_values, chosen)
    fig.suptitle(
        f"Sticky foresight breakdown  |  s={s}  |  choose {chosen}",
        fontsize=12,
    )
    return fig
