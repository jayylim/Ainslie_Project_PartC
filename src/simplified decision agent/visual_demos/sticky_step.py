'''
Stepped sticky sails + Bayesian stickiness belief panel.

Left: SS / LL sails and sticky aggregates under perceived ŝ.
Right: Beta posterior over stickiness s, with ŝ / fixed prior-mean overlay.

Stick/switch rolls and foresight both use perceived stickiness (feedback loop):
  learn Off → fixed preset = Beta prior mean α₀/(α₀+β₀)
  learn On  → updating Beta posterior mean (rolls use ŝ too)
Belief updates from stick/switch bits only. Intent does not override rolls.

Tune prior_alpha / prior_beta in main.sticky_env_params.

Run from the package root:

    python -m visual_demos.sticky_step
'''

import matplotlib
import matplotlib.pyplot as plt

import sys
from pathlib import Path

# Debugger / direct run put visual_demos/ on sys.
# path; package root is one up
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    matplotlib.use("MacOSX")
except Exception:
    pass

from agents.agent_configs import sticky_params
from agents.sticky_agent import StickyHypAgent
from environments.sticky_timeline import StickyTimelineEnvironment
from main import env_params, sticky_env_params, max_steps

# Agent sticky knobs: agents/agent_configs.py; Beta prior: main.sticky_env_params.
prior_alpha = sticky_env_params["prior_alpha"]
prior_beta = sticky_env_params["prior_beta"]
fixed_stickiness = prior_alpha / (prior_alpha + prior_beta)  # prior mean / preset
learn_stickiness = sticky_params["learn_stickiness"]
seed = None  # fully random; set integer for reproducible rolls

COLORS = {
    "dessert": "#c44e52",
    "resist": "#4c72b0",
    "dessert_sail": "#c44e52",
    "resist_sail": "#4c72b0",
    "belief": "#2ca02c",
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
        _, ss_outcomes = environment.simulate(state, "dessert")
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
        label = "dessert" if i == 0 else None
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
    """Sticky bundled dessert / resist values at the current decision (t=0)."""
    dessert_agg = action_values["dessert"]
    resist_agg = action_values["resist"]

    axes.scatter(
        0,
        dessert_agg,
        s=70,
        color=COLORS["dessert"],
        zorder=7,
        edgecolors=COLORS["dessert"],
        linewidths=1.2,
        label="dessert aggregate",
    )
    axes.text(
        -2,
        dessert_agg + 1,
        f"{dessert_agg:.2f}",
        color=COLORS["dessert"],
        va="bottom",
        ha="right",
        size=11,
        fontweight="bold" if chosen == "dessert" else "normal",
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

    return max(dessert_agg, resist_agg)


def plot_belief(ax_belief, beta_belief, perceived_s, *, learn):
    """Beta posterior density with perceived / fixed stickiness marker."""
    ax_belief.clear()
    xs, dens = beta_belief.pdf_grid(n=251)
    ax_belief.plot(xs, dens, color=COLORS["belief"], lw=2.0, label="Beta posterior")
    ax_belief.fill_between(xs, dens, color=COLORS["belief"], alpha=0.18)

    prior_mean = beta_belief.prior_mean()
    if learn:
        ax_belief.axvline(
            perceived_s,
            color=COLORS["perceived_s"],
            linestyle="-",
            lw=1.5,
            label=f"ŝ={perceived_s:.2f}",
        )
    else:
        ax_belief.axvline(
            prior_mean,
            color=COLORS["perceived_s"],
            linestyle="-",
            lw=1.5,
            label=f"fixed s=α/(α+β)={prior_mean:.2f}",
        )

    y_top = max(dens) if dens else 1.0
    ax_belief.set_xlim(0.0, 1.0)
    ax_belief.set_ylim(0.0, y_top * 1.15 if y_top else 1.0)
    ax_belief.set_xlabel("Stickiness s")
    ax_belief.set_ylabel("Density")
    ax_belief.set_title(
        f"Beta Distribution \n"
        f"α={beta_belief.alpha:.1f} | β={beta_belief.beta:.1f}\n"
        f"sticks={beta_belief.n_stick()} | switches={beta_belief.n_switch()}"
    )
    ax_belief.legend(loc="upper right", fontsize=9)


def draw_frame(ax_sails, ax_belief, agent, environment, beta_belief, state, steps):
    """Redraw sails + beta_belief from the current decision state."""
    ss_events, ll_events = outcome_events(agent, environment, state)
    n = agent.get_horizon(state, environment)
    s_hat = environment.s_hat()
    perceived_s = agent.perceived_stickiness(environment)
    prior_mean = beta_belief.prior_mean()

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

    y_max = max(y_max, plot_ss_sails(ax_sails, agent, ss_events, COLORS["dessert_sail"]))
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
    if learn_stickiness:
        s_line = f"Perceived stickiness={perceived_s:.3f} (learning)"
    else:
        s_line = (
            f"Fixed stickiness={prior_mean:.3f} "
            f"(prior mean α/(α+β), α={prior_alpha:g} β={prior_beta:g})"
        )
    ax_sails.set_title(
        f"Day {steps}  | {s_line}\n"
        f"horizon={n}  |  Agent would choose {intent}"
    )
    ax_sails.legend(loc="upper right")

    plot_belief(ax_belief, beta_belief, perceived_s, learn=learn_stickiness)

    dessert_days = agent.choice_horizon(state, "dessert", environment)
    resist_days = agent.choice_horizon(state, "resist", environment)

    print(
        f"Day {steps}  |  Horizon = {n}  ŝ = {s_hat:.4f}  "
        f"perceived_s = {perceived_s:.4f}  "
        f"fixed_s(prior mean) = {prior_mean:.4f}  "
        f"learn={learn_stickiness}"
    )
    print(
        f"  beta_belief α={beta_belief.alpha:.1f} β={beta_belief.beta:.1f}  "
        f"(prior α₀={beta_belief.alpha0:.1f} β₀={beta_belief.beta0:.1f})  "
        f"sticks={beta_belief.n_stick()} switches={beta_belief.n_switch()}"
    )
    print()
    print(
        "Table 1 — Pure hyperbolic day values\n"
        "  V_hyp(choice) from now at each decision node"
    )
    print(f"{'hours':>6}  {'stick_p':>10}  {'V_hyp(dessert)':>14}  {'V_hyp(resist)':>14}")
    for (t, _), (_, _) in zip(dessert_days, resist_days):
        hours = t * interval
        v_s = agent.day_value(state, "dessert", environment, t)
        v_r = agent.day_value(state, "resist", environment, t)
        print(
            f"{hours:6d}  {agent.stick_p_t(t, environment):10.4f}  "
            f"{v_s:14.4f}  {v_r:14.4f}"
        )
    print()
    print(
        "Table 2 — Sticky expected day values\n"
        "  EV_t | choice first = stick_p * V_hyp(a) + (1-stick_p) * V_hyp(b)"
    )
    print(f"{'hours':>6}  {'EV dessert-first':>16}  {'EV resist-first':>16}")
    for (t, ev_s), (_, ev_r) in zip(dessert_days, resist_days):
        hours = t * interval
        print(f"{hours:6d}  {ev_s:16.4f}  {ev_r:16.4f}")
    print(
        f"{'sum':>6}  {action_values['dessert']:16.6f}  {action_values['resist']:16.6f}"
    )
    print()
    print(f"Agent intent: {intent}")

    ll_ss_gap = action_values["resist"] - action_values["dessert"]
    print(f"LL−SS (sticky) at day 0: {ll_ss_gap:.4f}")
    print()

    return intent, ll_ss_gap


def main():
    environment = StickyTimelineEnvironment(
        **env_params,
        **sticky_env_params,
        seed=seed,
    )
    beta_belief = environment.beta_belief
    # Preset perceived_s = prior mean α/(α+β); used whenever learn_stickiness is Off
    agent = StickyHypAgent(
        **{**sticky_params, "perceived_s": fixed_stickiness}
    )
    state = environment.reset()

    finished = False
    steps = 0
    ll_ss_gaps = []

    fig, (ax_sails, ax_belief) = plt.subplots(
        1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [2.2, 1.0]}
    )
    print("Starting sticky + Bayesian visualisation (click / keypress to advance)...")
    print(
        f"learn_stickiness={learn_stickiness}  "
        f"prior α={prior_alpha} β={prior_beta}  "
        f"fixed_s=α/(α+β)={fixed_stickiness:.4f}  seed={seed}"
    )

    while not finished and (max_steps is None or steps < max_steps):
        steps += 1
        intent, ll_ss_gap = draw_frame(
            ax_sails, ax_belief, agent, environment, beta_belief, state, steps
        )
        ll_ss_gaps.append(ll_ss_gap)
        perceived_s = agent.perceived_stickiness(environment)
        state, finished, info = environment.step(intent, stickiness=perceived_s)
        align = "aligned" if info["aligned"] else "diverged"
        print(
            f"  intent={info['intent']}  realized={info['realized']}  "
            f"stuck={info['stuck']}  {align}  "
            f"roll_s={info['stickiness']:.4f}  New ŝ={info['s_hat']}"
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
