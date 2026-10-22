"""
Maximum SS that can still be resisted vs stickiness s, at decision node 1.

One curve per LL delay. LL reward is fixed (from env_params). Uses the
sticky_decisions model:

    expected_day_value = sticking * V_hyp(same) + (1 - sticking) * V_hyp(other)

With LL fixed and SS = x, set V(resist) = V(snack) and solve for x.
That x is both:
  - the maximum SS that can still be resisted (prefer LL at or below)
  - the minimum SS that makes the agent choose snack over LL

No environment rebuild: delay_LL is only a number in V_hyp.

Run from the package root:

    python -m visual_demos.sticky_graph
"""

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    matplotlib.use("MacOSX")
except Exception:
    pass

from agents.hyperbolic import HypAgent
from environments.timeline import TimelineEnvironment
from main import agent_params, env_params
from sticky_decisions import stick_p_t

# === Editable: LL delays to plot (hours after choice) ===
ll_delays = list(range(0, 25, 4))  # step size last number

# Stickiness sweep (for plotting various stickiness values)
stickiness_min, stickiness_max, n_stickiness_points = 0.0, 1.0, 100


def max_ss_resisted(agent, s, delay_SS, delay_LL, reward_LL, horizon, decision_interval):
    """
    Maximum SS such that V(resist) >= V(snack) under stickiness s (LL fixed).

    """
    # V(resist) = reward_LL * V_resist_LL + SS * V_resist_SS
    # V(snack)  = reward_LL * V_snack_LL  + SS * V_snack_SS
    V_resist_LL = 0.0
    V_resist_SS = 0.0
    V_snack_LL = 0.0
    V_snack_SS = 0.0

    for t in range(horizon):
        sticking = stick_p_t(s, t)
        # V_hyp(1, delay + t * decision_interval) — unit value on day t given delay

        # compute unit values
        v_SS = agent.value_function(1.0, delay_SS + t * decision_interval)
        v_LL = agent.value_function(1.0, delay_LL + t * decision_interval)

        # resist today: same = LL, other = SS
        V_resist_LL += sticking * v_LL
        V_resist_SS += (1.0 - sticking) * v_SS

        # snack today: same = SS, other = LL
        V_snack_SS += sticking * v_SS
        V_snack_LL += (1.0 - sticking) * v_LL

    # Set V(resist) = V(snack) with LL = reward_LL, SS = x:
    #   reward_LL * V_resist_LL + x * V_resist_SS
    #     = reward_LL * V_snack_LL + x * V_snack_SS
    #   x * (V_snack_SS - V_resist_SS) = reward_LL * (V_resist_LL - V_snack_LL)
    #   x = reward_LL * (V_resist_LL - V_snack_LL) / (V_snack_SS - V_resist_SS)
    solve_x = reward_LL * (V_resist_LL - V_snack_LL) / (V_snack_SS - V_resist_SS)
    return solve_x


environment = TimelineEnvironment(**env_params)
agent = HypAgent(**agent_params)
state = environment.reset()

horizon = agent.get_horizon(state, environment)
decision_interval = environment.decision_interval
discount_factor = agent.discount_factor
delay_SS = environment.rewards["SS"]["delay"]
reward_LL = environment.rewards["LL"]["value"]

# build x-axis values
stickiness_x = [
    stickiness_min
    + i * (stickiness_max - stickiness_min) / (n_stickiness_points - 1)
    for i in range(n_stickiness_points)
]

# One hue: short delays (top of chart) lightest / most transparent;
# longer delays darker / more opaque
base_rgb = matplotlib.colors.to_rgb("#1f4e79")
delay_lo, delay_hi = min(ll_delays), max(ll_delays)
delay_span = max(delay_hi - delay_lo, 1)

fig, axes = plt.subplots()
for delay_LL in ll_delays:
    # critical SS for this LL delay over the stickiness range
    max_ss_values = [
        max_ss_resisted(
            agent, s, delay_SS, delay_LL, reward_LL, horizon, decision_interval
        )
        for s in stickiness_x
    ]
    # 0 = shortest delay, 1 = longest
    t = (delay_LL - delay_lo) / delay_span
    # short delay → light + transparent; long delay → dark + opaque
    # lightness capped so short-delay lines stay visible on white
    lightness = 0.55 - 0.55 * t
    color = tuple(c + (1.0 - c) * lightness for c in base_rgb)
    # transparency capped at ~30% (alpha >= 0.70)
    alpha = 0.70 + 0.30 * t
    axes.plot(
        stickiness_x,
        max_ss_values,
        color=color,
        alpha=alpha,
        label=f"LL delay = {delay_LL}h",
    )

axes.set_xlabel("Stickiness s")
axes.set_ylabel("Max SS resisted (min SS to choose snack over resist)")
axes.set_title(
    f"Max SS resisted | "
    f"Current Settings: LL={reward_LL}, horizon={horizon}, discount_factor={discount_factor}"
) 
axes.set_xlim(stickiness_min, stickiness_max)
axes.set_ylim(0, 8)
axes.yaxis.set_major_locator(MultipleLocator(1))
axes.yaxis.set_minor_locator(MultipleLocator(0.25))
axes.grid(True, which="major", alpha=0.4)
axes.grid(True, which="minor", alpha=0.2)
axes.legend()

plt.show()

