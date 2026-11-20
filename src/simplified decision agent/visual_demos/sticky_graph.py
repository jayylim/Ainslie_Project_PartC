"""
Maximum SS that can still be resisted vs stickiness s, at decision node 1.

Curves for LL delays scaled from env_params (selected LL delay always
included). Set SHOW_OTHER_DELAYS = False to plot only that selected delay.
LL reward is fixed (from env_params). Uses the sticky_decisions model:

    expected_day_value = sticking * V_hyp(same) + (1 - sticking) * V_hyp(other)

With LL fixed and SS = x, set V(resist) = V(dessert) and solve for x.
That x is both:
  - the maximum SS that can still be resisted (prefer LL at or below)
  - the minimum SS that makes the agent choose dessert over LL

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

from agents.agent_configs import agent_params
from agents.hyperbolic import HypAgent
from environments.timeline import TimelineEnvironment
from main import env_params
from sticky_decisions import stick_p_t

# === Editable toggles / sweep ===
SHOW_OTHER_DELAYS = True  # False = only the env-selected LL delay curve

# Stickiness sweep (for plotting various stickiness values)
stickiness_min, stickiness_max, n_stickiness_points = 0.0, 1.0, 100


def max_ss_resisted(agent, s, delay_SS, delay_LL, reward_LL, horizon, decision_interval):
    """
    Maximum SS such that V(resist) >= V(dessert) under stickiness s (LL fixed).

    """
    # V(resist) = reward_LL * V_resist_LL + SS * V_resist_SS
    # V(dessert)  = reward_LL * V_dessert_LL  + SS * V_dessert_SS
    V_resist_LL = 0.0
    V_resist_SS = 0.0
    V_dessert_LL = 0.0
    V_dessert_SS = 0.0

    for t in range(horizon):
        sticking = stick_p_t(s, t)
        # V_hyp(1, delay + t * decision_interval) — unit value on day t given delay

        # compute unit values
        v_SS = agent.value_function(1.0, delay_SS + t * decision_interval)
        v_LL = agent.value_function(1.0, delay_LL + t * decision_interval)

        # resist today: same = LL, other = SS
        V_resist_LL += sticking * v_LL
        V_resist_SS += (1.0 - sticking) * v_SS

        # dessert today: same = SS, other = LL
        V_dessert_SS += sticking * v_SS
        V_dessert_LL += (1.0 - sticking) * v_LL

    # Set V(resist) = V(dessert) with LL = reward_LL, SS = x:
    #   reward_LL * V_resist_LL + x * V_resist_SS
    #     = reward_LL * V_dessert_LL + x * V_dessert_SS
    #   x * (V_dessert_SS - V_resist_SS) = reward_LL * (V_resist_LL - V_dessert_LL)
    #   x = reward_LL * (V_resist_LL - V_dessert_LL) / (V_dessert_SS - V_resist_SS)
    solve_x = reward_LL * (V_resist_LL - V_dessert_LL) / (V_dessert_SS - V_resist_SS)
    return solve_x


def ll_delays_for_env(decision_interval, main_delay_LL, show_other_delays):
    """
    Delay family scaled from the selected env: step ≈ interval/6, span one interval.
    Always includes the configured LL delay; other curves optional via toggle.
    """
    main_delay_LL = int(main_delay_LL)
    if not show_other_delays:
        return [main_delay_LL]

    delay_step = max(1, int(decision_interval) // 6)
    delay_max = max(int(decision_interval), main_delay_LL)
    # not including 0h as it is a straight line
    delays = list(range(delay_step, delay_max + 1, delay_step))
    if main_delay_LL > 0 and main_delay_LL not in delays:
        delays = sorted(delays + [main_delay_LL])
    return delays


environment = TimelineEnvironment(**env_params)
agent = HypAgent(**agent_params)
state = environment.reset()

horizon = agent.get_horizon(state, environment)
decision_interval = environment.decision_interval
discount_factor = agent.discount_factor
delay_SS = environment.rewards["SS"]["delay"]
reward_LL = environment.rewards["LL"]["value"]
main_delay_LL = int(environment.rewards["LL"]["delay"])
ll_delays = ll_delays_for_env(decision_interval, main_delay_LL, SHOW_OTHER_DELAYS)

# build x-axis values
stickiness_x = [
    stickiness_min
    + i * (stickiness_max - stickiness_min) / (n_stickiness_points - 1)
    for i in range(n_stickiness_points)
]

# One hue: short delays (top of chart) lightest / most transparent;
# longer delays darker / more opaque. Selected env delay drawn in black.
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
    is_main = delay_LL == main_delay_LL
    if is_main:
        color, alpha, linewidth = "#000000", 1.0, 2.8
        label = f"LL delay = {delay_LL}h"
    else:
        # 0 = shortest delay, 1 = longest
        t = (delay_LL - delay_lo) / delay_span
        # short delay → light + transparent; long delay → dark + opaque
        # lightness capped so short-delay lines stay visible on white
        lightness = 0.55 - 0.55 * t
        color = tuple(c + (1.0 - c) * lightness for c in base_rgb)
        # transparency capped at ~30% (alpha >= 0.70)
        alpha = 0.70 + 0.30 * t
        linewidth = 1.8
        label = f"LL delay = {delay_LL}h"
    axes.plot(
        stickiness_x,
        max_ss_values,
        color=color,
        alpha=alpha,
        linewidth=linewidth,
        label=label,
        zorder=3 if is_main else 2,
    )

axes.set_xlabel("Perceived stickiness (s)")
axes.set_ylabel("Max SS-value resisted)")
axes.set_title(
    "Max SS Resisted over Perceived Stickiness",
    pad=30,
)
axes.text(0.5, 1.02,
    f"LL value = {reward_LL} @ {main_delay_LL}h | "
    f"Bundling horizon = {horizon}, k = {discount_factor}",
    transform=axes.transAxes,
    ha="center",
    va="bottom",
    fontsize=10,
)
axes.set_xlim(stickiness_min, stickiness_max)
axes.set_ylim(0, 10)
axes.yaxis.set_major_locator(MultipleLocator(1))
axes.yaxis.set_minor_locator(MultipleLocator(0.25))
axes.grid(True, which="major", alpha=0.4)
axes.grid(True, which="minor", alpha=0.2)
axes.legend()

plt.show()

