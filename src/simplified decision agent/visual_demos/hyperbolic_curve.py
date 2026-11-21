"""
Very simplified visualisation of how a hyperbolically discounted reward looks
as the agent approaches its delivery time.

Run from the package root:

    python visual_demos/hyperbolic_curve.py
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

try:
    matplotlib.use("MacOSX")
except Exception:
    pass

LL = {
    "value": 10,
    "delay": 10,
}

discount = 0.8
N_SAMPLES = 200

fig, axes = plt.subplots()

x_max = LL["delay"] * 1.1
y_max = LL["value"] + 1

for current_delay in range(LL["delay"] + 1):
    remaining_delay = LL["delay"] - current_delay

    axes.clear()
    axes.set_xlim(-0.2, x_max)
    axes.set_ylim(0, y_max)
    axes.set_xlabel("Time from now (days)")
    axes.set_ylabel("Perceived value")
    axes.set_title(f"Choice at time step = {current_delay}")

    if remaining_delay <= 0:
        time_step = np.array([0.0])
        values = np.array([float(LL["value"])])
    else:
        time_step = np.linspace(0.0, remaining_delay, N_SAMPLES)
        distance = remaining_delay - time_step
        values = LL["value"] / (1.0 + discount * distance)

    axes.plot(time_step, values)
    axes.axvline(remaining_delay, linestyle="--")

    plt.pause(0.05)
    plt.waitforbuttonpress()

plt.show()
