"""
Shows the effective sail that the observer currently sees: the aggregate of
all discounted values across their vision horizon.

Matches HypAgent: each reward is discounted from the current decision, then
those values are summed (agents.hyperbolic.HypAgent.aggregate_value).

  - resist (LL): one curve where at each time t the height is the sum of
    V(reward_i, delay_i - t) over LLs still outstanding
  - snack (SS): one dot at t=0 with height = day-1 snack aggregate
    (same as action_values["snack"])

No stepping through the environment.

Run from the package root:

    python -m visual_demos.aggregate_sail
"""

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
from agents.agent_configs import agent_params
from main import env_params

COLORS = {"snack": "#c44e52", "resist": "#4c72b0"}


def horizon_events(agent, environment, state, action):
    """(delay, reward) pairs the agent counts for this action from now.

    short vision => one pair (instantaneous); long => full choice_horizon.
    """
    return agent.choice_horizon(state, action, environment)


def bundle_value_at(agent, events, at_time):
    """Sum of individually discounted rewards still outstanding at `at_time`."""
    return sum(
        agent.value_function(reward, delay - at_time)
        for delay, reward in events
        if delay >= at_time
    )


environment = TimelineEnvironment(**env_params)
agent = HypAgent(**agent_params)
state = environment.reset()

action, action_values = agent.choose_action(state, environment)

ss_events = horizon_events(agent, environment, state, "snack")
ll_events = horizon_events(agent, environment, state, "resist")

ll_delay = environment.rewards["LL"]["delay"]
# Sail height at t = discount-each-then-sum of LLs still left (agent rule)
ss_value = bundle_value_at(agent, ss_events, 0)  # == action_values["snack"] when long

fig, axes = plt.subplots()

xs, ys = [], []
for t in range(ll_delay + 1):
    xs.append(t)
    ys.append(bundle_value_at(agent, ll_events, t))

axes.plot(xs, ys, color=COLORS["resist"], label="resist")
axes.axvline(ll_delay, linestyle=":", color=COLORS["resist"], alpha=0.5)

perceived_ll = ys[0]
ll_tip = ys[-1]
axes.scatter(0, perceived_ll, s=50, color=COLORS["resist"], zorder=5)
axes.text(-1, perceived_ll, f"{perceived_ll:.2f}", color=COLORS["resist"],
          va="bottom", ha="right", size=12)
axes.scatter(ll_delay, ll_tip, s=50, color=COLORS["resist"], zorder=5)
axes.text(ll_delay, ll_tip, f"  {ll_tip:.2f}", color=COLORS["resist"],
          va="bottom", ha="left", size=12)

# SS: single dot at the day-1 aggregate (discount each SS, then sum)
axes.scatter(0, ss_value, s=50, color=COLORS["snack"], zorder=5, label="snack")
axes.text(-1, ss_value-2, f"{ss_value:.2f}", color=COLORS["snack"],

          va="bottom", ha="right", size=12)

y_max = max(max(ys) if ys else 0, ss_value)
axes.set_xlim(-max(2, ll_delay * 0.2), ll_delay + max(1, ll_delay * 0.15))
axes.set_ylim(0, y_max * 1.15 if y_max else 1)
axes.set_xlabel("Hours from now")
axes.set_ylabel("Perceived value")
axes.set_title(f"Bundled value on day 1 |  Agent would choose {action}")
axes.legend(loc="upper left")

print(f"Perceived values on day 1: snack={action_values['snack']:.4f}  "
      f"resist={action_values['resist']:.4f}")
print(f"Sail endpoints:            snack={ss_value:.4f}  "
      f"resist@0={perceived_ll:.4f}  resist@first LL={ll_tip:.4f}")
print(f"Would choose {action}")

plt.show()
