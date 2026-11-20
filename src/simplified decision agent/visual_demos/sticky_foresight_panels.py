'''
Optional sticky foresight panels (companion to visualise_sticky_sails).

Easy to reject later:
  1. Delete this file, and
  2. Remove the SHOW_STICKY_BREAKDOWN block / import from visualise_sticky_sails.py

Figure layout (one window, two panels):

  Left  — dessert-first choice trajectory (red fills)
  Right — resist-first choice trajectory (blue fills)

  Rows = foresight day t (0 at top). Two tracks per row: dessert | resist.
  Fill length = compounding visit probability under stickiness:
    P(a_t = a0 | a_0 = a0) = stick_p_t = 1/2 * (1 + (2s - 1)^t)
    P(other at t)          = 1 - stick_p_t
  Same panel colour for both tracks; only fill amount differs.
'''

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

COLORS = {
    "dessert": "#c44e52",
    "resist": "#4c72b0",
    "track": "#e8e8e8",
}

# Horizontal gap between dessert and resist tracks within a panel
COL_GAP = 0.08


def foresight_rows(agent, environment, state):
    """Visit-probability rows from StickyHypAgent.stick_p_t."""
    horizon = agent.get_horizon(state, environment)
    rows = []
    for t in range(horizon):
        stick_p = agent.stick_p_t(t)
        rows.append({"t": t, "stick_p": stick_p})
    return rows


def _track_probs(stick_p, a0):
    """P(visit dessert), P(visit resist) given today's candidate a0."""
    if a0 == "dessert":
        return stick_p, 1.0 - stick_p
    if a0 == "resist":
        return 1.0 - stick_p, stick_p
    raise ValueError(f"Unknown a0: {a0!r}")


def _label_ink(fill_color, p):
    """White text on dark fills, dark text on light fills."""
    r, g, b = mcolors.to_rgb(fill_color)
    # Perceived lightness of the filled segment blended over the track
    blend = 0.12 + 0.88 * p
    lum = 0.299 * (r * blend + (1 - blend)) + 0.587 * (g * blend + (1 - blend)) + 0.114 * (
        b * blend + (1 - blend)
    )
    return "#ffffff" if lum < 0.55 else "#222222"


# Beaker-style graduation marks along each track (fraction of full fill)
TICK_MINOR = (0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9)
TICK_MAJOR = (0.25, 0.5, 0.75)


def _draw_beaker_ticks(axes, y, track_left, track_right, bar_h, *, from_right=False):
    """Vertical measurement lines across one horizontal bar track."""
    half = bar_h / 2
    width = track_right - track_left

    def x_at(frac):
        # frac = filled fraction from the fill origin (left or right)
        if from_right:
            return track_right - frac * width
        return track_left + frac * width

    for frac in TICK_MINOR:
        x = x_at(frac)
        axes.plot(
            [x, x],
            [y - half * 0.35, y + half * 0.35],
            color="#666666",
            linewidth=0.6,
            solid_capstyle="butt",
            zorder=4,
            clip_on=True,
        )
    for frac in TICK_MAJOR:
        x = x_at(frac)
        axes.plot(
            [x, x],
            [y - half * 0.72, y + half * 0.72],
            color="#333333",
            linewidth=1.0,
            solid_capstyle="butt",
            zorder=4,
            clip_on=True,
        )


def plot_choice_trajectory(axes, rows, a0, color):
    """Horizontal fill bars: dessert | resist tracks, day 0 at top.

    Dessert fills left→right; resist fills right→left.
    Empty track to 1.0; filled segment length = visit probability; numeric label.
    Beaker-style ticks mark 0.1 steps (major at 0.25 / 0.5 / 0.75).
    Both tracks use `color` (same hue for the whole panel).
    """
    n = len(rows)
    if n == 0:
        return

    bar_h = 0.72
    resist_left = 1.0 + COL_GAP
    resist_right = resist_left + 1.0

    for row in rows:
        t = row["t"]
        p_dessert, p_resist = _track_probs(row["stick_p"], a0)

        # dessert track (column 0 → 1): fill from the left
        axes.barh(
            t,
            1.0,
            left=0.0,
            height=bar_h,
            color=COLORS["track"],
            edgecolor="white",
            linewidth=0.6,
            zorder=1,
        )
        axes.barh(
            t,
            p_dessert,
            left=0.0,
            height=bar_h,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            zorder=2,
        )
        _draw_beaker_ticks(axes, t, 0.0, 1.0, bar_h, from_right=False)
        axes.text(
            min(p_dessert, 0.98) if p_dessert >= 0.12 else p_dessert + 0.03,
            t,
            f"{p_dessert:.3f}",
            va="center",
            ha="right" if p_dessert >= 0.12 else "left",
            fontsize=9,
            color=_label_ink(color, p_dessert) if p_dessert >= 0.12 else "#333333",
            zorder=5,
        )

        # resist track (column resist_left → resist_right): fill from the right
        axes.barh(
            t,
            1.0,
            left=resist_left,
            height=bar_h,
            color=COLORS["track"],
            edgecolor="white",
            linewidth=0.6,
            zorder=1,
        )
        axes.barh(
            t,
            p_resist,
            left=resist_right - p_resist,
            height=bar_h,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            zorder=2,
        )
        _draw_beaker_ticks(axes, t, resist_left, resist_right, bar_h, from_right=True)
        if p_resist >= 0.12:
            label_x = resist_right - min(p_resist, 0.98)
            label_ha = "left"
            label_color = _label_ink(color, p_resist)
        else:
            label_x = resist_right - p_resist - 0.03
            label_ha = "right"
            label_color = "#333333"
        axes.text(
            label_x,
            t,
            f"{p_resist:.3f}",
            va="center",
            ha=label_ha,
            fontsize=9,
            color=label_color,
            zorder=5,
        )

    axes.set_xlim(-0.05, resist_right + 0.05)
    axes.set_ylim(n - 0.5, -0.5)  # day 0 at top
    axes.set_yticks(list(range(n)))
    axes.set_yticklabels([f"day {t}" for t in range(n)])
    axes.set_xticks([0.5, resist_left + 0.5])
    axes.set_xticklabels(["dessert", "resist"])
    axes.set_xlabel("action")
    axes.set_title(
        f"{a0}-first trajectory\n"
        r"fill $=P(\mathrm{visit})$ via $p_t=\frac{1}{2}(1+(2s-1)^t)$"
    )
    axes.set_axisbelow(True)
    axes.grid(False)
    for spine in ("top", "right"):
        axes.spines[spine].set_visible(False)


def show_sticky_breakdown(agent, environment, state, action_values, chosen, s):
    """Open twin choice-trajectory fill-bar panels (dessert-first | resist-first)."""
    rows = foresight_rows(agent, environment, state)
    fig, (ax_dessert, ax_resist) = plt.subplots(
        1, 2, figsize=(11, max(4.5, 0.45 * max(len(rows), 1) + 2.2)),
        constrained_layout=True,
    )
    plot_choice_trajectory(ax_dessert, rows, "dessert", COLORS["dessert"])
    plot_choice_trajectory(ax_resist, rows, "resist", COLORS["resist"])
    fig.suptitle(
        f"Sticky choice trajectories  |  s={s}  |  choose {chosen}",
        fontsize=12,
    )
    return fig
