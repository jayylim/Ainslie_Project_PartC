'''
Animate a full weighted sweep of snack-first sticky trajectories.

Bars start empty. Every snack-first path (2^(H-1)) is flashed once; fills
accumulate path probability mass and converge exactly to stick_p_t.

Refuses to run when horizon > 14 (too many paths to animate).

Run from the package root:

    python "animated demo/animate_trajectories.py"
'''

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

try:
    matplotlib.use("MacOSX")
except Exception:
    pass

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

from agents.sticky import StickyHypAgent
from environments.timeline import TimelineEnvironment
from main import agent_params, env_params

# === StickyHypAgent-related params (edit here) ===
stickiness = 0.6

# --- Demo knobs -----------------------------------------------------------
# Matplotlib can drop frames when INTERVAL_MS is tiny; update() catch-up
# commits any skipped batches so the final margins stay exact.
# Raise PATHS_PER_FRAME to go faster (more paths per redraw).
INTERVAL_MS = 2          # frame rate of animation
PATHS_PER_FRAME = agent_params["horizon"]/2  # paths per frame; flash only the last
RNG_SEED = 0              # shuffle enumeration order only
MAX_HORIZON = 14          # stop above this (2^(H-1) paths)
# ------------------------------------------------------------------------

COLORS = {
    "snack": "#c44e52",
    "track": "#e8e8e8",
    "target": "#888888",
    "highlight": "#111111",
}

COL_GAP = 0.08
BAR_H = 0.72
TICK_MINOR = (0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9)
TICK_MAJOR = (0.25, 0.5, 0.75)


def stick_p_t(s: float, t: int) -> float:
    """P(a_t = a | a_0 = a) = 1/2 * (1 + (2s - 1)^t)."""
    return 0.5 * (1.0 + (2.0 * s - 1.0) ** t)


def path_probability(path, s: float) -> float:
    """P(path | sticky chain) with day-0 fixed."""
    p = 1.0
    for t in range(1, len(path)):
        p *= s if path[t] == path[t - 1] else (1.0 - s)
    return p


def enumerate_snack_first_paths(horizon: int, s: float):
    """All snack-first trajectories and their sticky probabilities.

    Day 0 is snack; days 1..H-1 free → 2^(H-1) paths. Bits of the index
    choose snack (0) / resist (1) at each later day.
    """
    n_free = max(horizon - 1, 0)
    n_paths = 2**n_free
    paths = []
    probs = []
    for i in range(n_paths):
        path = ["snack"]
        for t in range(1, horizon):
            bit = (i >> (t - 1)) & 1
            path.append("resist" if bit else "snack")
        paths.append(path)
        probs.append(path_probability(path, s))
    return paths, np.asarray(probs, dtype=float)


def _label_ink(fill_color, p):
    r, g, b = mcolors.to_rgb(fill_color)
    blend = 0.12 + 0.88 * p
    lum = (
        0.299 * (r * blend + (1 - blend))
        + 0.587 * (g * blend + (1 - blend))
        + 0.114 * (b * blend + (1 - blend))
    )
    return "#ffffff" if lum < 0.55 else "#222222"


def _draw_beaker_ticks(axes, y, track_left, track_right, *, from_right=False):
    half = BAR_H / 2
    width = track_right - track_left

    def x_at(frac):
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
        )


class TrajectoryPanel:
    """Snack-first panel; fills = accumulated path-probability mass."""

    def __init__(self, axes, s: float, horizon: int, paths, probs):
        self.ax = axes
        self.a0 = "snack"
        self.color = COLORS["snack"]
        self.s = s
        self.horizon = horizon
        self.paths = paths
        self.probs = probs

        self.resist_left = 1.0 + COL_GAP
        self.resist_right = self.resist_left + 1.0

        self.mass_total = 0.0
        self.mass_snack = np.zeros(horizon, dtype=float)
        self.mass_resist = np.zeros(horizon, dtype=float)

        self.snack_fills = []
        self.resist_fills = []
        self.snack_labels = []
        self.resist_labels = []
        self.highlights = []
        self.current_path = None
        self.current_prob = 0.0

        self._setup_axes()
        self._draw_static()
        self._init_fills()
        self._init_highlights()

    def _setup_axes(self):
        n = self.horizon
        self.ax.set_xlim(-0.05, self.resist_right + 0.05)
        self.ax.set_ylim(n - 0.5, -0.5)
        self.ax.set_yticks(list(range(n)))
        self.ax.set_yticklabels([f"day {t}" for t in range(n)])
        self.ax.set_xticks([0.5, self.resist_left + 0.5])
        self.ax.set_xticklabels(["snack", "resist"])
        self.ax.set_xlabel("action")
        self.ax.set_title(
            "snack-first full sweep  |  weighted mass → "
            r"$p_t=\frac{1}{2}(1+(2s-1)^t)$"
        )
        self.ax.grid(False)
        for spine in ("top", "right"):
            self.ax.spines[spine].set_visible(False)

    def _draw_static(self):
        for t in range(self.horizon):
            self.ax.add_patch(
                Rectangle(
                    (0.0, t - BAR_H / 2),
                    1.0,
                    BAR_H,
                    facecolor=COLORS["track"],
                    edgecolor="white",
                    linewidth=0.6,
                    zorder=1,
                )
            )
            self.ax.add_patch(
                Rectangle(
                    (self.resist_left, t - BAR_H / 2),
                    1.0,
                    BAR_H,
                    facecolor=COLORS["track"],
                    edgecolor="white",
                    linewidth=0.6,
                    zorder=1,
                )
            )
            _draw_beaker_ticks(self.ax, t, 0.0, 1.0, from_right=False)
            _draw_beaker_ticks(
                self.ax, t, self.resist_left, self.resist_right, from_right=True
            )

            p_snack = stick_p_t(self.s, t)
            p_resist = 1.0 - p_snack
            self.ax.plot(
                [p_snack, p_snack],
                [t - BAR_H / 2, t + BAR_H / 2],
                color=COLORS["target"],
                linestyle="--",
                linewidth=1.0,
                alpha=0.7,
                zorder=3,
            )
            self.ax.plot(
                [self.resist_right - p_resist, self.resist_right - p_resist],
                [t - BAR_H / 2, t + BAR_H / 2],
                color=COLORS["target"],
                linestyle="--",
                linewidth=1.0,
                alpha=0.7,
                zorder=3,
            )

    def _init_fills(self):
        for t in range(self.horizon):
            snack = Rectangle(
                (0.0, t - BAR_H / 2),
                0.0,
                BAR_H,
                facecolor=self.color,
                edgecolor="white",
                linewidth=0.6,
                zorder=2,
            )
            resist = Rectangle(
                (self.resist_right, t - BAR_H / 2),
                0.0,
                BAR_H,
                facecolor=self.color,
                edgecolor="white",
                linewidth=0.6,
                zorder=2,
            )
            self.ax.add_patch(snack)
            self.ax.add_patch(resist)
            self.snack_fills.append(snack)
            self.resist_fills.append(resist)
            self.snack_labels.append(
                self.ax.text(
                    0.03,
                    t,
                    "0.000",
                    va="center",
                    ha="left",
                    fontsize=8,
                    color="#333333",
                    zorder=5,
                )
            )
            self.resist_labels.append(
                self.ax.text(
                    self.resist_right - 0.03,
                    t,
                    "0.000",
                    va="center",
                    ha="right",
                    fontsize=8,
                    color="#333333",
                    zorder=5,
                )
            )

    def _init_highlights(self):
        for t in range(self.horizon):
            rect = Rectangle(
                (0.0, t - BAR_H / 2),
                1.0,
                BAR_H,
                fill=False,
                edgecolor=COLORS["highlight"],
                linewidth=2.4,
                zorder=6,
                visible=False,
            )
            self.ax.add_patch(rect)
            self.highlights.append(rect)

    def _set_label(self, text_artist, p, *, from_right, track_right):
        y = text_artist.get_position()[1]
        if from_right:
            if p >= 0.12:
                text_artist.set_position((track_right - min(p, 0.98), y))
                text_artist.set_ha("left")
                text_artist.set_color(_label_ink(self.color, p))
            else:
                text_artist.set_position((track_right - p - 0.03, y))
                text_artist.set_ha("right")
                text_artist.set_color("#333333")
        else:
            if p >= 0.12:
                text_artist.set_position((min(p, 0.98), y))
                text_artist.set_ha("right")
                text_artist.set_color(_label_ink(self.color, p))
            else:
                text_artist.set_position((p + 0.03, y))
                text_artist.set_ha("left")
                text_artist.set_color("#333333")
        text_artist.set_text(f"{p:.3f}")

    def clear_highlights(self):
        for rect in self.highlights:
            rect.set_visible(False)

    def flash_path(self, path_index: int):
        path = self.paths[path_index]
        self.current_path = path
        self.current_prob = float(self.probs[path_index])
        for t, action in enumerate(path):
            rect = self.highlights[t]
            if action == "snack":
                rect.set_xy((0.0, t - BAR_H / 2))
            else:
                rect.set_xy((self.resist_left, t - BAR_H / 2))
            rect.set_width(1.0)
            rect.set_visible(True)

    def commit_path(self, path_index: int | None = None, *, refresh: bool = True, clear_highlight: bool = True):
        """Add this path's probability mass; optionally refresh fills."""
        if path_index is not None:
            self.current_path = self.paths[path_index]
            self.current_prob = float(self.probs[path_index])
        path = self.current_path
        w = self.current_prob
        for t, action in enumerate(path):
            if action == "snack":
                self.mass_snack[t] += w
            else:
                self.mass_resist[t] += w
        self.mass_total += w
        if refresh:
            self._refresh_fills()
        if clear_highlight:
            self.clear_highlights()

    def commit_batch(self, start: int, end: int):
        """Commit paths[start:end); flash only the last path in the batch."""
        if end <= start:
            return
        for i in range(start, end - 1):
            self.commit_path(i, refresh=False, clear_highlight=False)
        self.flash_path(end - 1)
        self.commit_path(refresh=True, clear_highlight=False)

    def empirical_margins(self):
        """Return (p_snack[t], p_resist[t]) from accumulated mass."""
        denom = self.mass_total if self.mass_total > 0 else 1.0
        snack = self.mass_snack / denom
        resist = self.mass_resist / denom
        return snack, resist

    def _refresh_fills(self):
        p_snack_all, p_resist_all = self.empirical_margins()
        for t in range(self.horizon):
            p_snack = float(p_snack_all[t])
            p_resist = float(p_resist_all[t])

            self.snack_fills[t].set_x(0.0)
            self.snack_fills[t].set_width(p_snack)

            self.resist_fills[t].set_x(self.resist_right - p_resist)
            self.resist_fills[t].set_width(p_resist)

            self._set_label(
                self.snack_labels[t],
                p_snack,
                from_right=False,
                track_right=1.0,
            )
            self._set_label(
                self.resist_labels[t],
                p_resist,
                from_right=True,
                track_right=self.resist_right,
            )


def build_batches(n_paths: int, paths_per_frame: int):
    """(start, end) index batches; each frame commits paths[start:end)."""
    step = max(int(paths_per_frame), 1)
    return [(i, min(i + step, n_paths)) for i in range(0, n_paths, step)]


def main():
    environment = TimelineEnvironment(**env_params)
    agent = StickyHypAgent(**agent_params, stickiness=stickiness)
    state = environment.reset()
    horizon = agent.get_horizon(state, environment)
    s = float(stickiness)

    n_all = 2**horizon
    n_snack_first = 2 ** max(horizon - 1, 0)
    print(f"Horizon H = {horizon}  stickiness s = {s}")
    print(f"True trajectories (any a0):     2^{horizon} = {n_all}")
    print(
        f"True trajectories (snack-first): 2^{max(horizon - 1, 0)} = {n_snack_first}  "
        "(day 0 fixed)"
    )

    if horizon > MAX_HORIZON:
        print(
            f"\nSTOP: horizon={horizon} exceeds MAX_HORIZON={MAX_HORIZON} "
            f"({n_snack_first} snack-first paths).\n"
            f"Lower agent horizon in main.py (or raise MAX_HORIZON) before "
            f"running the full weighted sweep."
        )
        sys.exit(1)

    n_redraws = max(1, (n_snack_first + PATHS_PER_FRAME - 1) // PATHS_PER_FRAME)
    print(f"Animating full weighted sweep of all {n_snack_first} snack-first paths.")
    print(
        f"Speed: INTERVAL_MS={INTERVAL_MS}, PATHS_PER_FRAME={PATHS_PER_FRAME} "
        f"(~{n_redraws} redraws). Raise PATHS_PER_FRAME to go faster."
    )

    paths, probs = enumerate_snack_first_paths(horizon, s)
    print(f"Sum of path probabilities = {float(probs.sum()):.6f} (expect ~1)")

    rng = np.random.default_rng(RNG_SEED)
    order = rng.permutation(len(paths))
    paths = [paths[i] for i in order]
    probs = probs[order]
    n_paths = len(paths)

    fig, ax = plt.subplots(
        figsize=(6.2, max(4.5, 0.45 * horizon + 2.2)),
        constrained_layout=True,
    )
    panel = TrajectoryPanel(ax, s, horizon, paths, probs)

    fig.suptitle(
        f"Sticky full sweep  |  s={s}  |  horizon={horizon}",
        fontsize=12,
    )
    progress = fig.text(
        0.5,
        0.01,
        f"0 / {n_paths}",
        ha="center",
        va="bottom",
        fontsize=10,
    )

    batches = build_batches(n_paths, PATHS_PER_FRAME)
    frames = batches + [None]
    # Catch-up cursor: if the GUI drops a frame, still commit every path once.
    committed = {"n": 0}

    def update_dispatch(frame):
        if frame is None:
            # Finish any remainder, then clear highlights.
            if committed["n"] < n_paths:
                panel.commit_batch(committed["n"], n_paths)
                committed["n"] = n_paths
            panel.clear_highlights()
            panel._refresh_fills()
            progress.set_text(f"{committed['n']} / {n_paths}")
            snack, _ = panel.empirical_margins()
            print("\nFinal empirical vs stick_p_t (should match):")
            for t in range(horizon):
                th = stick_p_t(s, t)
                print(
                    f"  day {t}: emp={snack[t]:.6f}  theory={th:.6f}  "
                    f"err={abs(snack[t] - th):.2e}"
                )
            return [
                *panel.snack_fills,
                *panel.resist_fills,
                *panel.snack_labels,
                *panel.resist_labels,
                *panel.highlights,
                progress,
            ]

        _start, end = frame
        # Commit from last position through this batch end (covers skipped frames).
        if end > committed["n"]:
            panel.commit_batch(committed["n"], end)
            committed["n"] = end
        progress.set_text(f"{committed['n']} / {n_paths}")
        return [
            *panel.snack_fills,
            *panel.resist_fills,
            *panel.snack_labels,
            *panel.resist_labels,
            *panel.highlights,
            progress,
        ]

    anim = FuncAnimation(
        fig,
        update_dispatch,
        frames=frames,
        interval=INTERVAL_MS,
        blit=False,
        repeat=False,
        cache_frame_data=False,
    )
    fig._sticky_traj_anim = anim  # noqa: SLF001  keep alive
    plt.show()


if __name__ == "__main__":
    main()
