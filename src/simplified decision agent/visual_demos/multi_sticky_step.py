'''
Multi-decision sticky step: aggregate panels, choice/stick grid, Beta belief.

Run from the package root:

    python -m visual_demos.multi_sticky_step
'''

import matplotlib
import matplotlib.pyplot as plt

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    matplotlib.use("MacOSX")
except Exception:
    pass

from agents.agent_configs import sticky_params
from agents.sticky_agent import StickyHypAgent
from environments.multi_sticky_timeline import MultiStickyTimelineEnvironment

env_params = {
    "length": None,
    "start": 0,
    "decision_interval": 24,
    "prior_alpha": 1.0,
    "prior_beta": 1.0,
    "decisions": {
        "Dessert": {
            "SS": {"value": 3, "delay": 0},
            "LL": {"value": 20, "delay": 12},
        },
        "Chips": {
            "SS": {"value": 2, "delay": 0},
            "LL": {"value": 18, "delay": 6},
        },
        "Soda": {
            "SS": {"value": 2, "delay": 0},
            "LL": {"value": 30, "delay": 18},
        },
    },
}

max_steps = 50
prior_alpha = env_params["prior_alpha"]
prior_beta = env_params["prior_beta"]
fixed_stickiness = prior_alpha / (prior_alpha + prior_beta)

COLORS = {"cheat": "#c44e52", "resist": "#4c72b0", "belief": "#2ca02c"}


def stuck_label(environment, decision_id, intent):
    last = environment.last_action.get(decision_id)
    if last is None:
        return "free"
    return "stick" if intent == last else "switch"

# visual of the decision(s) faced
def draw_aggregate_panel(ax, decision_id, intent, action_values, label):
    ax.clear()
    ss = action_values["cheat"]
    ll = action_values["resist"]
    ax.scatter(0, ss, s=110, color=COLORS["cheat"], zorder=5, label="SS")
    ax.scatter(0, ll, s=110, color=COLORS["resist"], zorder=5, label="LL")
    ax.text(
        0.08, ss, f"{ss:.2f}", color=COLORS["cheat"], va="center",
        fontweight="bold" if intent == "cheat" else "normal",
    )
    ax.text(
        0.08, ll, f"{ll:.2f}", color=COLORS["resist"], va="center",
        fontweight="bold" if intent == "resist" else "normal",
    )
    y_max = max(ss, ll)
    ax.set_xlim(-0.4, 1.0)
    ax.set_ylim(0, y_max * 1.25 if y_max else 1)
    ax.set_xticks([])
    ax.set_title(f"Decision: {decision_id} | Intent: {intent} ({label})", fontsize=11)
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=8, frameon=False)

# beta distribution for believed stickiness
def draw_belief(ax, beta_belief, perceived_s):
    ax.clear()
    xs, dens = beta_belief.pdf_grid(n=251)
    ax.plot(xs, dens, color=COLORS["belief"], lw=2.0)
    ax.fill_between(xs, dens, color=COLORS["belief"], alpha=0.18)
    ax.axvline(perceived_s, color="#ff7f0e", lw=1.5)
    y_top = max(dens) if dens else 1.0
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, y_top * 1.15 if y_top else 1.0)
    ax.set_xlabel("s")
    ax.set_title(
        f"Beta  α={beta_belief.alpha:.1f} β={beta_belief.beta:.1f}\n"
        f"ŝ={perceived_s:.3f}",
        fontsize=10,
    )

# line graph to track stickiness over time
def draw_stickiness(ax, s_history):
    ax.clear()
    xs = list(range(len(s_history)))
    ax.plot(xs, s_history, color="g", lw=2.0, marker="o", ms=5)
        
    window = max_steps / 4  
        #for better plot visual scaling
    # if len(s_history) > window:
    #     ax.set_xlim(len(s_history)-window, len(s_history))
    # else:
    #     ax.set_xlim(0, window)
        #alternative window config which just limits x
    ax.set_xticks(range(0, max_steps+1, 5))
    ax.set_xticks(range(0, max_steps+1), minor=True)
    ax.set_xlim(0.5, window)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Day")
    ax.set_ylabel("Stickiness s (prior mean)")
    if s_history:
        title = f"Stickiness over time\nnow={s_history[-1]:.3f}"
    ax.set_title(title, fontsize=10)
    ax.grid(True, axis="x", which = "major", alpha=0.3)


# a history grid of the agent's actions and sticky-statues for each decision type
def draw_history_grid(ax, decision_ids, history):
    ax.clear()
    n_ids = len(decision_ids)
    # Two rows per id; y decreases downward
    yticks = []
    ylabels = []
    for i, decision in enumerate(decision_ids):
        y_action = 2 * (n_ids - 1 - i) + 0.5
        y_stick = 2 * (n_ids - 1 - i)
        yticks.extend([y_action, y_stick])
        ylabels.extend([f"{decision} Choice", f"{decision} Stick/Switch"])

        actions = history[decision]["actions"]
        stucks = history[decision]["stuck"]
        for step_i, action in enumerate(actions):
            x = step_i + 1
            color = COLORS["cheat"] if action == "cheat" else COLORS["resist"]
            ax.scatter(x, y_action, s=70, color=color, zorder=3)
            stuck = stucks[step_i]
            if stuck is None:
                continue
            if stuck == 1:
                ax.scatter(x, y_stick, s=70, facecolors="none",
                           edgecolors="black", linewidths=1.4, zorder=3)
            else:
                ax.scatter(x, y_stick, s=70, marker="x", color="black",
                           linewidths=1.6, zorder=3)

    ax.set_ylim(-0.5, 2 * n_ids -1)
    ax.set_xlim(0.5, max_steps)
    ax.set_yticks(yticks)
    ax.set_xticks(range(0, max_steps+1, 5))
    ax.set_xticks(range(0, max_steps+1), minor=True)
    ax.set_yticklabels(ylabels, fontsize=9)
    ax.set_xlabel("Day")
    ax.set_title(
        f"Record of Decisions and Stick/Switch\n"
        f"SS/LL (red/blue)  ·  stick ○ / switch ×", fontsize=10)
    ax.grid(True, axis="x", which = "major", alpha=0.2)


def draw_frame(ax_by_id, ax_grid, ax_belief, ax_s, agent, environment, state, history, s_history):
    intents, values = agent.choose_all_actions(state, environment)
    perceived_s = agent.perceived_stickiness(environment)
    for decision in environment.decision_ids:
        draw_aggregate_panel(
            ax_by_id[decision],
            decision,
            intents[decision],
            values[decision],
            stuck_label(environment, decision, intents[decision]),
        )
    draw_history_grid(ax_grid, environment.decision_ids, history)
    draw_belief(ax_belief, environment.beta_belief, perceived_s)
    draw_stickiness(ax_s, s_history)
    return intents


def main():
    environment = MultiStickyTimelineEnvironment(**env_params)
    agent = StickyHypAgent(
        **{**sticky_params, "perceived_s": fixed_stickiness}
    )
    state = environment.reset()
    finished = False
    steps = 0
    ids = environment.decision_ids
    history = {decision: {"actions": [], "stuck": []} for decision in ids}
    s_history = [environment.s_hat()]  # prior mean at step 0

    n = len(ids)
    fig = plt.figure(figsize=(4.0 * n + 3.5, 7.5))
    gs = fig.add_gridspec(
        2, n + 1,
        width_ratios=[1.0] * n + [0.85],
        height_ratios=[1.2, 1.5],
        hspace=0.35,
        wspace=0.35,
    )
    ax_by_id = {decision: fig.add_subplot(gs[0, i]) for i, decision in enumerate(ids)}
    ax_grid = fig.add_subplot(gs[1, :n])
    ax_belief = fig.add_subplot(gs[0, n])
    ax_s = fig.add_subplot(gs[1, n])

    while not finished and (max_steps is None or steps < max_steps):
        steps += 1
        # Panels = prediction for this upcoming step; grid = prior realizations only
        intents = draw_frame(
            ax_by_id,
            ax_grid,
            ax_belief,
            ax_s,
            agent,
            environment,
            state,
            history,
            s_history,
        )
        plt.tight_layout()
        plt.pause(0.05)
        plt.waitforbuttonpress()

        state, finished, info = environment.step(intents)
        for decision in ids:
            dinfo = info["per_decision"][decision]
            history[decision]["actions"].append(dinfo["realized"])
            history[decision]["stuck"].append(dinfo["stuck"])
        s_history.append(info["s_hat"])
        print(f"  committed  ŝ={info['s_hat']:.3f}")

    print("Done.")
    plt.show()


if __name__ == "__main__":
    main()
