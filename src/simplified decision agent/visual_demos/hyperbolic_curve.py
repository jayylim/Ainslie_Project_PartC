"""
Very simplified visualisation of how a hyperbolically discounted reward looks
as the agent approaches its delivery time.

Run from the package root:

    python visual_demos/hyperbolic_curve.py
"""

import matplotlib.pyplot as plt
import matplotlib

try:
    matplotlib.use("MacOSX")
except Exception:
    pass

LL = {
    "value": 10,
    "delay": 10,
}

discount = 0.8

fig, axes = plt.subplots()

x_max = LL["delay"]*1.1
y_max = LL["value"] + 1

for current_delay in range(LL["delay"] + 1):
    remaining_delay = LL["delay"] - current_delay

    axes.clear()
    axes.set_xlim(-0.2, x_max)
    axes.set_ylim(0, y_max)
    axes.set_xlabel("Time to reward (days)")
    axes.set_ylabel("Perceived value")
    axes.set_title(f"Choice at time step = {current_delay}")

    values = []
    time_step = []

    perceived_value = LL["value"] / (1 + discount * remaining_delay)

    for t in range(remaining_delay + 1):
        distance = remaining_delay - t
        value = LL["value"] / (1 + discount * distance)
        time_step.append(t)
        values.append(value)

    axes.plot(time_step, values)
    axes.axvline(remaining_delay, linestyle="--")
    #axes.scatter(0, perceived_value, s=50, color="black")
    #axes.text(0, perceived_value + 0.5, round(perceived_value, 2), ha="center", size=14)

    plt.pause(0.05)
    plt.waitforbuttonpress()

plt.show()
