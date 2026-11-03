"""
Run a hyperbolic (or sticky hyperbolic) agent on the memoryless SS/LL timeline.
Edit the params below, then run from this directory:

    python main.py

Units are hours. One choice per day (decision_interval=24).
LL delay is hours after the choice (e.g. 12 ≈ next morning).

Set agent_kind = "sticky" to use StickyHypAgent (stickiness in agent_params).
Set agent_kind = "hyperbolic" for the original pure-commitment HypAgent.

To visualise value curves at each decision point:

    python -m visual_demos.visualize_main
"""

from agents.hyperbolic import HypAgent
from agents.sticky import StickyHypAgent
from environments.timeline import TimelineEnvironment

# === Params ===
env_params = {
    "length": None,  # Time in hours; or None for infinite
    "start": 0,
    "decision_interval": 24,  # hours between decisions
    "rewards": {
        "SS": {"value": 3, "delay": 0},   # snack now
        "LL": {"value": 20, "delay": 12},  # hours after choice until weigh-in
    },
}

# "hyperbolic" => HypAgent (pure commitment when vision=long)
# "sticky"     => StickyHypAgent (foresight under stickiness)
agent_kind = "hyperbolic" 

agent_params = {
    "discount_factor": 0.6,
    "vision": "long",  # "short" or "long"
    "horizon": 10,  # max # of decisions to plan; capped by remaining env decisions, includes the current day (min. 1)
}

# used by StickyHypAgent only; s=1 matches HypAgent long vision
stickiness = 0.6
max_steps = None  # used when env length is None; also caps the run as a safety stop


def make_agent():
    if agent_kind == "hyperbolic":
        return HypAgent(**agent_params)
    if agent_kind == "sticky":
        return StickyHypAgent(**agent_params, stickiness=stickiness)
    raise ValueError(f"Unknown agent_kind: {agent_kind!r}")


def run():
    environment = TimelineEnvironment(**env_params)
    agent = make_agent()

    state = environment.reset()
    finished = False
    steps = 0

    print("Starting...")
    print(f"state={state}h (interval={environment.decision_interval}h)")
    print(f"agent_kind={agent_kind}")
    if agent_kind == "sticky":
        print(f"stickiness={agent.stickiness}")

    while not finished and steps < max_steps:
        action, action_values = agent.choose_action(state, environment)
        # earned_reward = environment.payout(action) ## only if tracking payouts
        state, finished = environment.step(action)
        steps += 1

        print(f"Day {steps}: options={action_values}")
        print(f"  Chose {action}")

    print("Done.")


if __name__ == "__main__":
    run()
