"""
Compare exponential vs hyperbolic discounting for one reward (LL).

Exponential:  V = reward * (discount ** distance)
Hyperbolic:   V = reward / (1 + discount * distance)

Run from the package root:

    python visual_demos/exponential_or_hyperbolic.py
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

try:
    matplotlib.use("MacOSX")
except Exception:
    pass

# SS = {
#     "value": 10,
#     "delay": 3,
# }

LL = {
    "value": 30,
    "delay": 10,
}

discount = 0.8  # δ / k

COLORS = {
    "exponential": "#c44e52",
    "hyperbolic": "#4c72b0",
}

fig, axes = plt.subplots()

y_max = LL["value"] + 1
plot_x_max = 19
N_SAMPLES = 200

axes.set_xlim(0, plot_x_max)
axes.set_ylim(0, y_max)
#axes.set_xlabel("Hours from now")
axes.set_xlabel("Hours from reward")
axes.set_ylabel("Reward value")
axes.set_xticks(np.arange(0, plot_x_max+1, 2.5))
#axes.set_ylabel("Perceived value")
#axes.set_title("Exponential vs Hyperbolic Discounting (LL)")
axes.set_title(f"Exponential Discounting of Rewards Over Time")
#axes.set_title(f"Exponentially Increasing Value of Approaching Rewards")

# time_step = []
# values = []
# for t in range(plot_x_max):
# #for t in range(remaining + 1):
#     #distance = remaining - t
#     distance = t
#     value = reward["value"] * (discount ** remaining)
#     # time_step.append(t)
#     # values.append(value)
time_step = np.linspace(0.0, plot_x_max, N_SAMPLES)
distance = time_step

values_exp = LL["value"] * (discount ** distance)  # exponential
values_hyp = LL["value"] / (1.0 + discount * distance)  # hyperbolic

axes.plot(
    time_step,
    values_exp,
    color=COLORS["exponential"],
    label=f'Exponential Discounting',
)
# axes.plot(
#     time_step,
#     values_hyp,
#     color=COLORS["hyperbolic"],
#     label=f'Hyperbolic Discounting',
# )
#axes.axvline(remaining, linestyle="--", color=color, alpha=0.5)

#axes.legend(loc="upper right")
plt.show()
