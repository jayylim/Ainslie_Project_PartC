"""
Very simplified visualisation of hyperbolic discounting for two rewards
(smaller-sooner vs larger-later) as they approach an agent fixed at t=0.

V(reward, distance) = reward / (1 + k * distance)

As time elapses the remaining delay shortens, so each sail rises toward
full reward at delivery. Click / press a key to step forward one hour.

Run from the package root:

    python visual_demos/hyperbolic_ss_ll.py
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

try:
    matplotlib.use("MacOSX")
except Exception:
    pass

SS = {
    "value": 3,
    "delay": 16,
}

LL = {
    "value": 15,
    "delay": 24,
}

discount = 0.8  # k in V = reward / (1 + k * distance)
N_SAMPLES = 200

COLORS = {
    "SS": "#c44e52",
    "LL": "#4c72b0",
}

fig, axes = plt.subplots()

x_max = max(SS["delay"], LL["delay"])
y_max = max(SS["value"], LL["value"]) + 1


def perceived(reward, remaining):
    return reward["value"] / (1.0 + discount * remaining)


for current_time in range(x_max + 1):
    axes.clear()
    axes.set_xlim(-0.2, x_max+1)
    axes.set_ylim(0, y_max)
    axes.set_xlabel("Hours to approaching reward")
    axes.set_ylabel("Perceived value")

    rem_ss = SS["delay"] - current_time
    rem_ll = LL["delay"] - current_time
    v_ss = perceived(SS, rem_ss) if rem_ss >= 0 else None
    v_ll = perceived(LL, rem_ll) if rem_ll >= 0 else None

    if v_ss is not None and v_ll is not None:
        if abs(v_ss - v_ll) < 1e-12:
            pref = "indifferent"
        elif v_ss > v_ll:
            pref = "prefers SS"
        else:
            pref = "prefers LL"
    elif v_ss is not None:
        pref = "SS only (LL delivered)"
    elif v_ll is not None:
        pref = "LL only (SS delivered)"
    else:
        pref = "both delivered"

    axes.set_title(
        f"Rewards approaching | Time to LL = {LL['delay'] - current_time}h  |  {pref}"
    )

    for label, reward, remaining, v_now in (
        ("SS", SS, rem_ss, v_ss),
        ("LL", LL, rem_ll, v_ll),
    ):
        if remaining < 0:
            continue

        color = COLORS[label]

        if remaining <= 0:
            time_step = np.array([0.0])
            values = np.array([float(reward["value"])])
        else:
            time_step = np.linspace(0.0, remaining, N_SAMPLES)
            distance = remaining - time_step
            values = reward["value"] / (1.0 + discount * distance)

        axes.plot(
            time_step,
            values,
            color=color,
            label=f'{label}: {reward["value"]} in {reward["delay"]}h',
        )
        axes.axvline(remaining, linestyle="--", color=color, alpha=0.5)
        axes.scatter([0], [v_now], color=color, zorder=5)
        axes.scatter([remaining], [reward["value"]], color=color, zorder=5)
        # axes.annotate(
        #     f"{label} {v_now:.2f}",
        #     (0, v_now),
        #     textcoords="offset points",
        #     xytext=(6, 6),
        #     color=color,
        #     fontsize=9,
        # )

    #axes.legend(loc="upper right")
    plt.pause(0.05)
    plt.waitforbuttonpress('q')

plt.show()
