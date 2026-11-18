"""
Very simplified visualisation of exponential discounting for two rewards
(smaller-sooner vs larger-later) as the agent approaches delivery.

V(reward, delay) = reward * (discount ** delay)

Run from the package root:

    python visual_demos/exponential_curve.py
"""

import matplotlib.pyplot as plt
import matplotlib

try:
    matplotlib.use("MacOSX")
except Exception:
    pass

SS = {
    "value": 10,
    "delay": 10,
}

LL = {
    "value": 30,
    "delay": 12,
}

discount = 0.8  # δ in V = reward * δ^delay

COLORS = {
    "SS": "#c44e52",
    "LL": "#4c72b0",
}

fig, axes = plt.subplots()

x_max = max(SS["delay"], LL["delay"])
y_max = max(SS["value"], LL["value"]) + 1

for current_time in range(x_max + 1):
    axes.clear()
    axes.set_xlim(-0.2, x_max)
    axes.set_ylim(0, y_max)
    axes.set_xlabel("Hours from now")
    axes.set_ylabel("Perceived value")
    axes.set_title(f"Exponentially Increasing Value of Approaching Rewards")

    for label, reward in (("SS", SS), ("LL", LL)):
        remaining = reward["delay"] - current_time
        if remaining < 0:
            continue

        color = COLORS[label]
        time_step = []
        values = []
        for t in range(remaining + 1):
            distance = remaining - t
            value = reward["value"] * (discount ** distance)
            time_step.append(t)
            values.append(value)

        perceived = reward["value"] * (discount ** remaining)
        axes.plot(
            time_step,
            values,
            color=color,
            label=f'{label}: {reward["value"]} in {reward["delay"]} hours',
        )
        axes.axvline(remaining, linestyle="--", color=color, alpha=0.5)

    
    axes.legend(loc="upper right")
    plt.pause(0.05)
    plt.waitforbuttonpress()

plt.show()
