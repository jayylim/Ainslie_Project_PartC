"""
Factoring in stickiness into the aggregated decision.

Shows the agent's computed values for the first decision node.

Snapshot compputation of t=1: Does not step the environment. 
Reuses main.py, hyperbolic.py, and timeline.py params and helpers.

Definitions and notation (see legend at bottom of this docstring):

    s   = P(a_{t+1} = a_t)                         stickiness
    stick_p_t = 1/2 + 1/2 * (2s - 1)^t                   persistence of today's action (collapses all possible trees where you arrive here)
    V_hyp(reward, delay) = reward / (1 + discount_factor * delay)
    day_t value = stick_p_t * V_hyp(reward_a, delay_a + t in days) + (1-stick_p_t) * V_hyp(reward_b, delay_b + t in days) # the value of ending on said action at day_t
    V(action) = sum_{t=0}^{horizon-1} [
             stick_p_t     * V_hyp(reward_a, delay_a + t * decision_interval)
           + (1-stick_p_t) * V_hyp(reward_b, delay_b + t * decision_interval)
           ]                                       sticky bundle value

When s = 1, V(a) matches the demonstration in main.py (pure commitment).

Run from the package root:

    python -m sticky_decisions

Legend
------
s                  stickiness; P(next decision equals current decision)
a                  candidate action being valued today (snack or resist)
b                  the other action (not a)
a_t                action on foresight day t (random under stickiness)
t                  foresight day index, t = 0 .. horizon-1 (0 = today)
sticking           probability that on day t, the agent makes the same choice as day 0
horizon            # of decision nodes from get_horizon (short vision => 1)
decision_interval  hours between choice nodes (Delta)
discount_factor    k in V_hyp
reward_a, delay_a  reward magnitude and base delay for action a's hyp_value
reward_b, delay_b  same for action b
persistence        P(a_t = a | a_0 = a); derived from s
V_hyp              hyperbolic discounted value of one (reward, delay) pair
V(a)               expected bundled value of choosing a today under s
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.hyperbolic import HypAgent
from environments.timeline import TimelineEnvironment
from main import agent_params, env_params

# === Stickiness s (edit here) ===
s = 0.8  # in [0, 1]; s = 1 recovers pure long-vision bundling

OTHER = {"snack": "resist", "resist": "snack"}  # b given a


def stick_p_t(s, t):
    # P(a_t = a | a_0 = a) = 1/2 *(1 + (2s - 1)^t)
    return 0.5 * (1 + (2 * s - 1) ** t)

# have to make a separate valuation method because agent's method automatically aggregates 
def hyp_value(agent, environment, state, action, t):
    # Hyperbolic value of `action`'s value delayed by t decision intervals.
    _, outcomes = environment.simulate(state, action)
    decision_interval = environment.decision_interval
    total = 0.0
    for reward, delay in outcomes.values():
        total += agent.value_function(
            reward, delay + t * decision_interval
        )  # V_hyp(reward, delay + t*decision_interval)
    return total


def value_table(agent, environment, state, s):
    """
    Shared foresight table (independent of today's candidate):
    stick_p_t and V_hyp(snack), V_hyp(resist) at each t.
    """
    horizon = agent.get_horizon(state, environment)
    rows = []
    for t in range(horizon):
        rows.append(
            {
                "t": t,
                "stick_p_t": stick_p_t(s, t),
                "v_snack": hyp_value(agent, environment, state, "snack", t),
                "v_resist": hyp_value(agent, environment, state, "resist", t),
            }
        )
    return rows


def V_from_table(table, a):
    """
    Sticky bundle value V(a) from the shared value table.
    a is today's candidate; b is the other action.
    EV_t = stick_p_t * V_hyp(a) + (1 - stick_p_t) * V_hyp(b)
    """
    b = OTHER[a]
    v_key = {"snack": "v_snack", "resist": "v_resist"}
    total = 0.0
    breakdown = []
    for row in table:
        sticking = row["stick_p_t"]
        v_same = row[v_key[a]]
        v_other = row[v_key[b]]
        expected_day_value = sticking * v_same + (1 - sticking) * v_other
        total += expected_day_value
        breakdown.append(
            {
                "t": row["t"],
                "expected_day_value": expected_day_value,
            }
        )
    return total, breakdown


def greedy_from_values(action_values):
    return max(action_values, key=action_values.get)


def main():
    environment = TimelineEnvironment(**env_params)
    agent = HypAgent(**agent_params)
    state = environment.reset()
    horizon = agent.get_horizon(state, environment)
    decision_interval = environment.decision_interval
    discount_factor = agent.discount_factor

    print(f"state={state}h  horizon={horizon}  stickiness={s}")
    print(f"decision_interval={decision_interval}h  discount_factor={discount_factor}")
    print()

    # --- 1. Generate shared value table ---
    table = value_table(agent, environment, state, s)
    print(
        "Table 1 — Discounted values from today\n"
        " V_hyp(choice) = value_choice / (1 + discount_factor * delay)\n"
        " stick_p_t = P(a_t = a | a_0 = a) = 1/2 *(1 + (2s - 1)^t)"
    )
    print(f"{'t':>3}  {'stick_p_t':>10}  {'V_hyp(snack)':>14}  {'V_hyp(resist)':>14}")
    for row in table:
        print(
            f"{row['t']:3d}  {row['stick_p_t']:10.4f}  {row['v_snack']:14.4f}  "
            f"{row['v_resist']:14.4f}"
        )
    print()

    # --- 2. Expected day value under stickiness (both branches side by side) ---
    V_snack, snack_rows = V_from_table(table, "snack")
    V_resist, resist_rows = V_from_table(table, "resist")
    V_values = {"snack": V_snack, "resist": V_resist}

    print(
        "Table 2 — Expected day value by branch\n"
        "  EV_t | choice first = stick_p_t * V_hyp(choice) + (1-stick_p_t) * V_hyp(other)\n"
    )
    print(f"{'t':>3}  {'snack first':>14}  {'resist first':>14}")
    for snack_row, resist_row in zip(snack_rows, resist_rows):
        print(
            f"{snack_row['t']:3d}  {snack_row['expected_day_value']:14.4f}  "
            f"{resist_row['expected_day_value']:14.4f}"
        )
    print(f"{'sum':>3}  {V_snack:14.6f}  {V_resist:14.6f}")
    print()

    # --- 3. Sticky choice vs s=1 baseline ---
    choice = greedy_from_values(V_values)
    print(f"Sticky agent (s={s}) would choose {choice}")
    print()

    baseline_action, baseline_values = agent.choose_action(state, environment)
    print(
        "Baseline (s=1, pure commitment via HypAgent.aggregate_value): "
        "assumes today's action is repeated every foresight day."
    )
    print(
        f"  snack={baseline_values['snack']:.6f}  "
        f"resist={baseline_values['resist']:.6f}"
    )
    print(f"Baseline would choose {baseline_action}")

    if abs(s - 1) < 1e-12:
        ok = all(
            abs(V_values[a] - baseline_values[a]) < 1e-9 for a in V_values
        )
        print(
            f"Sanity s=1 match: {'PASS' if ok else 'FAIL'} "
            f"(V(a) vs HypAgent.choose_action)"
        )


if __name__ == "__main__":
    main()
