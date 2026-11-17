'''
Stepped sticky sails + Bayesian stickiness belief panel.

Left: SS / LL sails and sticky aggregates under perceived ŝ.
Right: Beta posterior over stickiness s, with s_true and ŝ overlays.

Environment: StickyTimelineEnvironment rolls stick/switch from true_stickiness.
Agent intent is logged; realized action follows the roll (day 0 = free intent).
Belief updates from stick/switch bits only; s_true never changes.

When learn_stickiness is True, StickyHypAgent foresight uses belief.mean() as ŝ.

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
from environments.sticky_timeline import StickyTimelineEnvironment
from learning.bayesian_stickiness import BayesianStickinessBelief
from main import agent_params, env_params, max_steps

# === StickyHypAgent / Bayesian env params (edit here) ===
stickiness = 0.6          # initial fixed ŝ when learn_stickiness is False
true_stickiness = 0.8     # generative s_true (never updated by learning)
learn_stickiness = True   # True => foresight uses belief.mean() as ŝ
prior_alpha = 1.0         # Beta prior on s
prior_beta = 1.0
seed = 0
# Shared timeline / hyperbolic knobs still come from main.env_params / main.agent_params.

COLORS = {
    "snack": "#c44e52",
    "resist": "#4c72b0",
    "snack_sail": "#c44e52",
    "resist_sail": "#4c72b0",
    "belief": "#2ca02c",
    "true_s": "#111111",
    "perceived_s": "#ff7f0e",
}


def fade(base, i, n, max_fade=0.72):
    """i=0 full base colour; i=n-1 faded toward white. Scales with horizon n."""
    rgb = matplotlib.colors.to_rgb(base)
    t = (i / (n - 1) if n > 1 else 0.0) * max_fade
    return tuple(c + (1.0 - c) * t for c in rgb)


def outcome_events(agent, environment, state):
    """(delay, reward) SS / LL pairs at each foresight day."""
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
    """SS sails feed back to t=0."""
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


def plot_belief(ax_belief, belief, true_s, perceived_s):
    """Beta posterior density with true and perceived stickiness markers."""
    ax_belief.clear()
    xs, dens = belief.pdf_grid(n=251)
    ax_belief.plot(xs, dens, color=COLORS["belief"], lw=2.0, label="Beta posterior")
    ax_belief.fill_between(xs, dens, color=COLORS["belief"], alpha=0.18)

    ax_belief.axvline(
        true_s,
        color=COLORS["true_s"],
        linestyle="--",
        lw=1.5,
        label=f"s_true={true_s:.2f}",
    )
    ax_belief.axvline(
        perceived_s,
        color=COLORS["perceived_s"],
        linestyle="-",
        lw=1.5,
        label=f"ŝ={perceived_s:.2f}",
    )

    y_top = max(dens) if dens else 1.0
    ax_belief.set_xlim(0.0, 1.0)
    ax_belief.set_ylim(0.0, y_top * 1.15 if y_top else 1.0)
    ax_belief.set_xlabel("Stickiness s")
    ax_belief.set_ylabel("Density")
    ax_belief.set_title(
        f"Belief  α={belief.alpha:.1f}  β={belief.beta:.1f}  |  "
        f"sticks={belief.n_stick()}  switches={belief.n_switch()}"
    )
    ax_belief.legend(loc="upper right", fontsize=9)


def draw_frame(ax_sails, ax_belief, agent, environment, belief, state, steps):
    """Redraw sails + belief from the current decision state."""
    ss_events, ll_events = outcome_events(agent, environment, state)
    n = agent.get_horizon(state, environment)
    s_hat = agent.perceived_stickiness()

    intent, action_values = agent.choose_action(state, environment)

    ax_sails.clear()
    y_max = 0

    n_sails = len(ll_events)
    for i, (delay, reward) in enumerate(ll_events):
        decision_t = ss_events[i][0]
        sail_color = fade(COLORS["resist_sail"], i, n_sails)
        peak = plot_ll_sail(
            ax_sails,
            agent,
            delay,
            reward,
            decision_t,
            sail_color,
            label_base=COLORS["resist_sail"],
            label="resist" if i == 0 else None,
        )
        y_max = max(y_max, peak)

    y_max = max(y_max, plot_ss_sails(ax_sails, agent, ss_events, COLORS["snack_sail"]))
    y_max = max(y_max, plot_aggregates(ax_sails, action_values, intent))

    interval = environment.decision_interval
    horizon_days = agent.horizon if agent.horizon is not None else n
    x_max = horizon_days * interval
    pad_x = max(8, x_max * 0.12)
    ax_sails.set_xlim(-pad_x, x_max + pad_x)
    ax_sails.set_ylim(0, y_max * 1.25 if y_max else 1)

    xticks = list(range(0, x_max + 1, 12))
    ax_sails.set_xticks(xticks)
    ax_sails.set_xticklabels([str(t) for t in xticks])
    ax_sails.set_xlabel("Hours from now")
    ax_sails.set_ylabel("Perceived value")
    ax_sails.set_title(
        f"Day {steps}  |  ŝ={s_hat:.3f}  s_true={true_stickiness}  |  "
        f"horizon={n}  |  intent={intent}"
    )
    ax_sails.legend(loc="upper right")

    plot_belief(ax_belief, belief, true_stickiness, s_hat)

    snack_days = agent.choice_horizon(state, "snack", environment)
    resist_days = agent.choice_horizon(state, "resist", environment)

    print(
        f"Day {steps}  |  Horizon = {n}  ŝ = {s_hat:.4f}  "
        f"s_true = {true_stickiness}  learn={learn_stickiness}"
    )
    print(
        f"  belief α={belief.alpha:.1f} β={belief.beta:.1f}  "
        f"sticks={belief.n_stick()} switches={belief.n_switch()}"
    )
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
    print(f"Agent intent: {intent}")

    ll_ss_gap = action_values["resist"] - action_values["snack"]
    print(f"LL−SS (sticky) at day 0: {ll_ss_gap:.4f}")
    print()

    return intent, ll_ss_gap


def main():
    belief = BayesianStickinessBelief(alpha=prior_alpha, beta=prior_beta)
    environment = StickyTimelineEnvironment(
        **env_params,
        true_stickiness=true_stickiness,
        belief=belief,
        seed=seed,
    )
    agent = StickyHypAgent(
        **agent_params,
        stickiness=stickiness,
        stickiness_belief=belief,
        learn_stickiness=learn_stickiness,
    )
    state = environment.reset()

    finished = False
    steps = 0
    ll_ss_gaps = []

    fig, (ax_sails, ax_belief) = plt.subplots(
        1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [2.2, 1.0]}
    )
    print("Starting sticky + Bayesian visualisation (click / keypress to advance)...")
    print(f"s_true={true_stickiness}  learn_stickiness={learn_stickiness}  seed={seed}")

    while not finished and (max_steps is None or steps < max_steps):
        steps += 1
        intent, ll_ss_gap = draw_frame(
            ax_sails, ax_belief, agent, environment, belief, state, steps
        )
        ll_ss_gaps.append(ll_ss_gap)
        state, finished, info = environment.step(intent)
        print(
            f"  realized={info['realized']}  stuck={info['stuck']}  "
            f"ŝ_after={info['perceived_mean']}"
        )
        print()
        plt.tight_layout()
        plt.pause(0.05)
        plt.waitforbuttonpress()

    print(f"Day {steps} LL−SS gaps: {ll_ss_gaps}")
    print(f"stick_history={environment.stick_history}")
    print(f"intent_history={environment.intent_history}")
    print(f"action_history={environment.action_history}")
    print("Done.")
    plt.show()


if __name__ == "__main__":
    main()
