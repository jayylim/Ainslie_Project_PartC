"""
Scan sticky-step learning dynamics over prior mean s.

Same algorithm as the web app sticky_step demo:
  - StickyTimelineEnvironment + StickyHypAgent(learn_stickiness=True)
  - prior α, β = lowest positive integers with mean s  (web_app.prior_from_mean)
  - each step: greedy intent under current ŝ → realize intent → Beta update
    from stick/switch vs previous action (no update on the first day)

Reports contiguous ranges of s for:
  1. lock_in_rising_s   — preference never flips; ŝ is nondecreasing (sticks forever)
  2. one_shot_switch    — exactly one preference flip, then locked on the new action
  3. switching_loop     — more than one preference flip (with pattern subtype)

Run from the package root:

    python scan_sticky_learning.py

Edit the knobs below (same style as main.py).
"""

from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

_PKG = Path(__file__).resolve().parent
sys.path.insert(0, str(_PKG))
sys.path.insert(0, str(_PKG / "web_app"))

from agents.agent_configs import sticky_params
from agents.sticky_agent import StickyHypAgent
from environments.sticky_timeline import StickyTimelineEnvironment
from main import env_params
from prior_from_mean import integers_for_mean

# === Defaults shared with web Scan tab ===
DEFAULT_SCAN_OPTS = {
    "n_steps": 60,
    "s_min": 0.5,
    "s_max": 0.99,
    "s_step": 0.01,
    # View filter only (N=α+β); does not change integer prior construction
    "strength_min": 2,
    "strength_max": 200,
}

# === Scan knobs (CLI / script entrypoint) ===
n_steps = 200          # decision days per prior s
s_min = 0.5
s_max = 0.99
s_step = 0.01
# None → use sticky_params
discount_factor = None
horizon = None
# Figure only: keep rows whose integer prior strength N=α+β is in [lo, hi]
# (does not change how priors are built — still lowest integers for each mean)
strength_min = DEFAULT_SCAN_OPTS["strength_min"]
strength_max = DEFAULT_SCAN_OPTS["strength_max"]
# Figure
show_plot = True
save_path = None  # e.g. "sticky_scan.png"

ACTION_COLORS = {"dessert": "#c44e52", "resist": "#4c72b0"}
LABEL_COLOR = {
    "lock_in_rising_s": "#2ca02c",
    "lock_in": "#7f7f7f",
    "switching_loop": "#d62728",
}


def row_marker_color(row):
    """Scatter / strip colour: one-shot uses the action switched to; else regime colour."""
    if row.get("label") == "one_shot_switch":
        return ACTION_COLORS.get(row.get("last"), "#ff7f0e")
    return LABEL_COLOR.get(row.get("label"), "#cccccc")


def prior_strength(row):
    """N = α + β for an integer prior row."""
    return int(row["alpha"]) + int(row["beta"])


def filter_rows_by_strength(rows, strength_min=None, strength_max=None):
    """
    View filter only: keep rows with strength_min ≤ α+β ≤ strength_max.
    Does not alter prior construction (priors stay lowest integers for each mean).
    """
    lo = 0 if strength_min is None else int(strength_min)
    hi = 10**9 if strength_max is None else int(strength_max)
    if lo > hi:
        lo, hi = hi, lo
    return [r for r in rows if lo <= prior_strength(r) <= hi]


# ---------------------------------------------------------------------------
# Simulation (mirrors web_app sticky_step step loop)
# ---------------------------------------------------------------------------


@dataclass
class Trajectory:
    s0: float
    alpha0: int
    beta0: int
    actions: list
    means: list  # ŝ after each step, index 0 = prior mean before any choice
    values: list  # (dessert, resist) aggregates at each decision
    stuck: list  # None on first day, then 0/1


def simulate(s, *, n_steps, agent_kwargs, env_kwargs):
    alpha, beta, achieved = integers_for_mean(s)
    environment = StickyTimelineEnvironment(
        **env_kwargs,
        prior_alpha=alpha,
        prior_beta=beta,
    )
    agent = StickyHypAgent(
        **{
            **agent_kwargs,
            "perceived_s": achieved,
            "learn_stickiness": True,
        }
    )
    state = environment.reset()
    traj = Trajectory(
        s0=achieved,
        alpha0=alpha,
        beta0=beta,
        actions=[],
        means=[environment.s_hat()],
        values=[],
        stuck=[],
    )
    for _ in range(n_steps):
        intent, action_values = agent.choose_action(state, environment)
        traj.values.append(
            (float(action_values["dessert"]), float(action_values["resist"]))
        )
        state, _finished, info = environment.step(intent)
        traj.actions.append(intent)
        traj.means.append(environment.s_hat())
        traj.stuck.append(info["stuck"])
    return traj


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def _flip_indices(actions):
    return [i for i in range(1, len(actions)) if actions[i] != actions[i - 1]]


def _run_lengths(actions):
    if not actions:
        return []
    runs = []
    cur, n = actions[0], 1
    for a in actions[1:]:
        if a == cur:
            n += 1
        else:
            runs.append((cur, n))
            cur, n = a, 1
    runs.append((cur, n))
    return runs


def _means_nondecreasing(means, eps=1e-15):
    """True if ŝ never falls (first day is flat: no Beta update yet)."""
    return all(means[i + 1] + eps >= means[i] for i in range(len(means) - 1))


def _loop_subtype(actions, flips):
    """Describe ongoing preference switching more finely."""
    if len(actions) < 4 or len(flips) < 2:
        return "sparse_flips"

    # Late window pattern (ignore a short burn-in).
    burn = min(5, len(actions) // 5)
    late = actions[burn:]
    runs = _run_lengths(late)
    if not runs:
        return "sparse_flips"

    lengths = [n for _, n in runs]
    majority = Counter(a for a, _ in runs).most_common(1)[0][0]
    minority_runs = [n for a, n in runs if a != majority]
    majority_runs = [n for a, n in runs if a == majority]

    # Strict alternation DRDR… / RDRD…
    if all(n == 1 for n in lengths) and len(runs) >= 4:
        return "period_2_alternation"

    # Long majority runs punctuated by single (or short) minority spikes.
    if (
        minority_runs
        and majority_runs
        and max(minority_runs) <= 2
        and min(majority_runs) >= 3
        and sum(1 for n in minority_runs if n == 1) >= max(1, len(minority_runs) // 2)
    ):
        return f"intermittent_{majority}_with_spikes"

    gaps = [flips[i] - flips[i - 1] for i in range(1, len(flips))]
    if gaps and max(gaps) - min(gaps) <= 2 and min(gaps) >= 2:
        return f"quasi_periodic_period_{round(sum(gaps) / len(gaps))}"

    return "irregular_switching"


@dataclass
class Classification:
    label: str
    detail: str = ""
    n_flips: int = 0
    first_action: str | None = None
    last_action: str | None = None
    s_start: float = 0.0
    s_end: float = 0.0
    means_rising: bool = False
    flip_days: list = field(default_factory=list)


def classify(traj: Trajectory) -> Classification:
    actions = traj.actions
    means = traj.means
    flips = _flip_indices(actions)
    rising = _means_nondecreasing(means)
    first = actions[0] if actions else None
    last = actions[-1] if actions else None
    base = dict(
        n_flips=len(flips),
        first_action=first,
        last_action=last,
        s_start=means[0],
        s_end=means[-1],
        means_rising=rising,
        flip_days=[i + 1 for i in flips],  # 1-based decision day of the new action
    )

    if not actions:
        return Classification(label="empty", **base)

    if len(flips) == 0:
        return Classification(
            label="lock_in_rising_s" if rising else "lock_in",
            detail=f"always {first}",
            **base,
        )

    # Exactly one flip in the whole horizon → permanent preference change.
    if len(flips) == 1:
        return Classification(
            label="one_shot_switch",
            detail=f"{first} → {last} at day {flips[0] + 1}",
            **base,
        )

    subtype = _loop_subtype(actions, flips)
    return Classification(
        label="switching_loop",
        detail=subtype,
        **base,
    )


# ---------------------------------------------------------------------------
# Range aggregation + report
# ---------------------------------------------------------------------------


def contiguous_ranges(rows, key):
    """Merge consecutive scan rows with the same key into (s_lo, s_hi, rows)."""
    if not rows:
        return []
    out = []
    start = 0
    for i in range(1, len(rows) + 1):
        if i == len(rows) or key(rows[i]) != key(rows[start]):
            chunk = rows[start:i]
            out.append((chunk[0]["s"], chunk[-1]["s"], chunk))
            start = i
    return out


def contiguous_where(rows, predicate, key=None):
    """Contiguous stretches on the full s-grid where predicate(row) is true."""
    if key is None:
        key = lambda _r: True
    out = []
    i = 0
    while i < len(rows):
        if not predicate(rows[i]):
            i += 1
            continue
        j = i + 1
        k0 = key(rows[i])
        while j < len(rows) and predicate(rows[j]) and key(rows[j]) == k0:
            j += 1
        chunk = rows[i:j]
        out.append((chunk[0]["s"], chunk[-1]["s"], chunk))
        i = j
    return out


def action_string(actions, max_len=72):
    s = "".join("D" if a == "dessert" else "R" for a in actions)
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


def unique_s_targets(s_min, s_max, s_step):
    """Grid of prior means, deduped by integer (α, β) from integers_for_mean."""
    if s_step <= 0:
        raise ValueError("s_step must be positive")
    if not (0.0 < s_min < s_max < 1.0):
        raise ValueError("need 0 < s_min < s_max < 1")
    n = int(round((s_max - s_min) / s_step))
    s_values = [s_min + i * s_step for i in range(n + 1)]
    s_values = [s for s in s_values if 0.0 < s < 1.0]
    seen = set()
    unique = []
    for s in s_values:
        alpha, beta, achieved = integers_for_mean(s)
        key = (alpha, beta)
        if key in seen:
            continue
        seen.add(key)
        unique.append(achieved)
    return unique


def q1_q2_summary(rows, *, n_steps):
    """Structured Q1 / Q2 text for the web dialog (no Q3)."""
    lines = [
        f"steps={n_steps}  points={len(rows)}  "
        f"s in [{rows[0]['s']:.4f}, {rows[-1]['s']:.4f}]",
        "",
        "Q1 — s continually increases (preference is locked-in / all sticks)",
        f"  (Finite horizon: some mid-s locks (especailly to SS) may only mean no switches yet within {n_steps} steps.)",
    ]
    q1 = contiguous_where(
        rows,
        lambda r: r["label"] == "lock_in_rising_s",
        key=lambda r: r["first"],
    )
    if not q1:
        lines.append("  (none in this grid)")
    else:
        for s_lo, s_hi, chunk in q1:
            lines.append(
                f"  s ∈ [{s_lo:.4f}, {s_hi:.4f}]  always {chunk[0]['first']}  "
                f"ŝ_end in [{min(c['s_end'] for c in chunk):.3f}, "
                f"{max(c['s_end'] for c in chunk):.3f}]"
            )

    lines.extend(["", "Q2 — one preference switch, never switches back"])
    one_shot = [r for r in rows if r["label"] == "one_shot_switch"]
    if not one_shot:
        lines.append("  (none in this grid / horizon)")
    else:
        for r in one_shot:
            lines.append(
                f"  s={r['s']:.4f}  α={r['alpha']} β={r['beta']}  "
                f"{r['detail']}  ŝ_end={r['s_end']:.3f}"
            )
            lines.append(f"    {action_string(r['actions'])}")

    return {
        "text": "\n".join(lines),
        "q1_ranges": [
            {
                "s_lo": s_lo,
                "s_hi": s_hi,
                "first": chunk[0]["first"],
                "n": len(chunk),
            }
            for s_lo, s_hi, chunk in q1
        ],
        "q2_one_shot": [
            {
                "s": r["s"],
                "alpha": r["alpha"],
                "beta": r["beta"],
                "detail": r["detail"],
                "s_end": r["s_end"],
                "last": r["last"],
            }
            for r in one_shot
        ],
    }


def build_scan(
    s_values,
    *,
    n_steps,
    agent_kwargs,
    env_kwargs,
):
    rows = []
    for s in s_values:
        traj = simulate(
            s, n_steps=n_steps, agent_kwargs=agent_kwargs, env_kwargs=env_kwargs
        )
        cls = classify(traj)
        rows.append(
            {
                "s": traj.s0,
                "target_s": float(s),
                "alpha": traj.alpha0,
                "beta": traj.beta0,
                "n": int(traj.alpha0) + int(traj.beta0),
                "label": cls.label,
                "detail": cls.detail,
                "n_flips": cls.n_flips,
                "means_rising": cls.means_rising,
                "s_end": cls.s_end,
                "first": cls.first_action,
                "last": cls.last_action,
                "flip_days": cls.flip_days,
                "actions": traj.actions,
                "means": traj.means,
            }
        )
    return rows


def print_report(rows, *, n_steps):
    print("=" * 72)
    print("Sticky learning regime scan")
    print(
        f"steps={n_steps}  "
        f"points={len(rows)}  "
        f"s in [{rows[0]['s']:.4f}, {rows[-1]['s']:.4f}]"
    )
    print("=" * 72)

    def regime_key(r):
        # Keep loop subtypes separate so ranges stay meaningful.
        if r["label"] == "switching_loop":
            return (r["label"], r["detail"])
        if r["label"] in ("lock_in_rising_s", "lock_in"):
            return (r["label"], r["first"])
        return (r["label"], r["detail"])

    print("\n--- Contiguous ranges ---")
    for s_lo, s_hi, chunk in contiguous_ranges(rows, regime_key):
        label, detail = regime_key(chunk[0])
        detail_bit = f" · {detail}" if detail else ""
        print(
            f"  s ∈ [{s_lo:.4f}, {s_hi:.4f}]  "
            f"({len(chunk)} pts)  {label}{detail_bit}"
        )

    # Focused answers to the three scan questions (ranges respect grid gaps).
    one_shot = [r for r in rows if r["label"] == "one_shot_switch"]

    print("\n--- Q1: ŝ continually increases (preference lock-in / all sticks) ---")
    print(
        "  (Finite horizon: some mid-s 'locks' only mean no flip yet within "
        f"{n_steps} steps; raise --steps to re-check.)"
    )
    q1 = contiguous_where(
        rows,
        lambda r: r["label"] == "lock_in_rising_s",
        key=lambda r: r["first"],
    )
    if not q1:
        print("  (none in this grid)")
    else:
        for s_lo, s_hi, chunk in q1:
            print(
                f"  s ∈ [{s_lo:.4f}, {s_hi:.4f}]  always {chunk[0]['first']}  "
                f"ŝ_end in [{min(c['s_end'] for c in chunk):.3f}, "
                f"{max(c['s_end'] for c in chunk):.3f}]"
            )

    print("\n--- Q2: one preference switch, never switches back ---")
    if not one_shot:
        print("  (none in this grid / horizon)")
    else:
        for r in one_shot:
            print(
                f"  s={r['s']:.4f}  α={r['alpha']} β={r['beta']}  "
                f"{r['detail']}  ŝ_end={r['s_end']:.3f}"
            )
            print(f"    {action_string(r['actions'])}")

    print("\n--- Per-point summary ---")
    print(
        f"{'s':>7}  {'α':>4} {'β':>4}  {'flips':>5}  {'rising':>6}  "
        f"{'ŝ_end':>6}  label"
    )
    for r in rows:
        print(
            f"{r['s']:7.4f}  {r['alpha']:4d} {r['beta']:4d}  "
            f"{r['n_flips']:5d}  {str(r['means_rising']):>6}  "
            f"{r['s_end']:6.3f}  {r['label']}"
            + (f" ({r['detail']})" if r["detail"] else "")
        )


def _s_edges(s_values):
    if len(s_values) == 1:
        s = s_values[0]
        return [s - 0.01, s + 0.01]
    mid = [0.5 * (s_values[i] + s_values[i + 1]) for i in range(len(s_values) - 1)]
    return (
        [s_values[0] - (mid[0] - s_values[0])]
        + mid
        + [s_values[-1] + (s_values[-1] - mid[-1])]
    )

# === Plotting ===

def plot_scan(
    rows,
    *,
    n_steps,
    strength_min=None,
    strength_max=None,
    save_path=None,
    show=True,
):
    """Action raster (day × s); left edge coloured by regime.

    strength_min / strength_max filter the view by prior N=α+β only.
    (Web Scan tab shows the prior Beta PDF for a clicked row.)
    """
    rows = filter_rows_by_strength(rows, strength_min, strength_max)
    if not rows:
        print("No rows left after prior-strength filter; widen strength_min/max.")
        return

    s_vals = [r["s"] for r in rows]
    n_rows = len(rows)
    grid = np.array(
        [[0 if a == "dessert" else 1 for a in r["actions"]] for r in rows]
    )
    # Equal-height rows (index axis); tick labels still show prior mean s.
    y_edges = np.arange(n_rows + 1) - 0.5
    x_edges = np.arange(0.5, n_steps + 1.5)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.pcolormesh(
        x_edges,
        y_edges,
        grid,
        cmap=ListedColormap([ACTION_COLORS["dessert"], ACTION_COLORS["resist"]]),
        vmin=0,
        vmax=1,
        shading="flat",
    )
    for r, y0, y1 in zip(rows, y_edges[:-1], y_edges[1:]):
        ax.axhspan(
            y0,
            y1,
            xmin=0,
            xmax=0.02,
            color=row_marker_color(r),
            zorder=5,
        )

    ax.set_xlabel("Decision day")
    ax.set_ylabel("Prior mean s")
    ax.set_title("Action over time (red=dessert, blue=resist)")
    ax.set_xlim(0.5, n_steps + 0.5)
    ax.set_ylim(y_edges[0], y_edges[-1])
    tick_step = 1 if n_rows <= 40 else max(1, int(round(n_rows / 30)))
    tick_idx = list(range(0, n_rows, tick_step))
    ax.set_yticks(tick_idx)
    ax.set_yticklabels([f"{s_vals[i]:.3f}" for i in tick_idx])
    ax.legend(
        handles=[
            Patch(color=ACTION_COLORS["dessert"], label="dessert / one-shot→dessert"),
            Patch(color=ACTION_COLORS["resist"], label="resist / one-shot→resist"),
            Patch(color=LABEL_COLOR["lock_in_rising_s"], label="lock (ŝ↑)"),
            Patch(color=LABEL_COLOR["switching_loop"], label="loop"),
        ],
        loc="upper right",
        fontsize=8,
    )
    n_lo = min(prior_strength(r) for r in rows)
    n_hi = max(prior_strength(r) for r in rows)
    fig.suptitle(
        f"Sticky learning scan  (view filter N=α+β ∈ [{n_lo}, {n_hi}])",
        fontsize=13,
    )
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved figure → {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    if s_step <= 0:
        raise SystemExit("s_step must be positive")
    if not (0.0 < s_min < s_max < 1.0):
        raise SystemExit("need 0 < s_min < s_max < 1")

    unique_targets = unique_s_targets(s_min, s_max, s_step)

    agent_kwargs = {
        "discount_factor": (
            sticky_params["discount_factor"]
            if discount_factor is None
            else discount_factor
        ),
        "vision": sticky_params["vision"],
        "horizon": sticky_params["horizon"] if horizon is None else horizon,
    }
    env_kwargs = {**env_params, "length": None}

    rows = build_scan(
        unique_targets,
        n_steps=n_steps,
        agent_kwargs=agent_kwargs,
        env_kwargs=env_kwargs,
    )
    print_report(rows, n_steps=n_steps)
    plot_scan(
        rows,
        n_steps=n_steps,
        strength_min=strength_min,
        strength_max=strength_max,
        save_path=save_path,
        show=show_plot,
    )


if __name__ == "__main__":
    main()
