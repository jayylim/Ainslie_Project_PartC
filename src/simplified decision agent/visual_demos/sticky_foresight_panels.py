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
          Side Σ bars are sectioned by each foresight EV_t and labeled
          with hours + contribution; totals are V(a) = sum_t EV_t(a).
          Σ segment colours fade dark→light with foresight (modular in n).
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
    x_max = max(hours) if hours else 0
    xticks = list(range(0, int(x_max) + 1, 24))
    axes.set_xticks(xticks)
    axes.set_xticklabels([str(t) for t in xticks])
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

    # Aggregate Σ bars: sectioned by each foresight day's EV_t contribution
    total_x0 = hours[-1] + group_gap * 1.35
    x_sum_snack = total_x0 - offset
    x_sum_resist = total_x0 + offset
    ev_snack = np.array([r["ev_snack"] for r in rows])
    ev_resist = np.array([r["ev_resist"] for r in rows])
    v_snack = float(np.sum(ev_snack))
    v_resist = float(np.sum(ev_resist))
    # Prefer agent aggregates when provided (should match)
    if action_values is not None:
        v_snack = action_values["snack"]
        v_resist = action_values["resist"]

    snack_cmap = plt.cm.Reds
    resist_cmap = plt.cm.Blues

    def fade_cmap(cmap, i, n_seg):
        # i=0 dark, i=n-1 light; scales with horizon
        t = i / (n_seg - 1) if n_seg > 1 else 0.0
        return cmap(0.88 - 0.56 * t)

    bottom_s = 0.0
    bottom_r = 0.0
    min_label = 0.12 * max(v_snack, v_resist, 1e-9)  # skip tiny segment labels
    for i, r in enumerate(rows):
        h_s = ev_snack[i]
        h_r = ev_resist[i]
        edge_s = 1.2 if chosen == "snack" else 0.5
        edge_r = 1.2 if chosen == "resist" else 0.5
        t = i / (n - 1) if n > 1 else 0.0
        ink = "#ffffff" if t < 0.55 else "#333333"
        axes.bar(
            x_sum_snack,
            h_s,
            width=bar_w * 1.15,
            bottom=bottom_s,
            color=fade_cmap(snack_cmap, i, n),
            edgecolor="black",
            linewidth=edge_s,
            label=r"$\sum EV_t$ snack (near→far, dark→light)" if i == 0 else None,
        )
        axes.bar(
            x_sum_resist,
            h_r,
            width=bar_w * 1.15,
            bottom=bottom_r,
            color=fade_cmap(resist_cmap, i, n),
            edgecolor="black",
            linewidth=edge_r,
            label=r"$\sum EV_t$ resist (near→far, dark→light)" if i == 0 else None,
        )
        # Label each section by foresight hour + contribution
        if h_s >= min_label:
            axes.text(
                x_sum_snack,
                bottom_s + h_s / 2,
                f"{int(r['hours'])}h\n{h_s:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color=ink,
            )
        if h_r >= min_label:
            axes.text(
                x_sum_resist,
                bottom_r + h_r / 2,
                f"{int(r['hours'])}h\n{h_r:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color=ink,
            )
        bottom_s += h_s
        bottom_r += h_r

    axes.text(
        total_x0,
        max(v_snack, v_resist) * 1.04,
        f"Σ snack={v_snack:.2f}  |  Σ resist={v_resist:.2f}",
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
    )

    axes.set_xticks(list(hours) + [total_x0])
    axes.set_xticklabels([str(int(h)) for h in hours] + ["Σ"])
    axes.set_xlabel("Hours from now")
    axes.set_ylabel("Value contribution (from now)")
    axes.set_title(
        "Stacked day contributions\n"
        r"hourly bars $=EV_t$; Σ bars sectioned by each $EV_t$ → $V(a)$"
    )
    axes.legend(loc="upper left", fontsize=7)
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
