"""
Run a hyperbolic agent on the memoryless SS/LL timeline.
Edit env params below; agent init defaults live in agents/agent_configs.py.
Units are hours. One choice per day (decision_interval=24).
LL delay is hours after the choice (e.g. 12 ≈ next morning).

Run from this directory:

    python main.py

"""

from agents.agent_configs import agent_params
from agents.hyperbolic import HypAgent
from environments.timeline import TimelineEnvironment

# === Env params (agent init: agents/agent_configs.py) ===
env_params = {
    "length": None,  # Time in hours; or None for infinite
    "start": 0,
    "decision_interval": 24,  # hours between decisions
    "rewards": {
        "SS": {"value": 3, "delay": 0},   # snack now
        "LL": {"value": 20, "delay": 12},  # hours after choice until weigh-in
    },
    # StickyTimelineEnvironment only (generative s_true; never learned)
    "true_stickiness": 0.8,
}

max_steps = None  # used when env length is None to cap this run as a safety stop. Should always be none for steppable infinite environments.


def make_agent():
    return HypAgent(**agent_params)


def run():
    environment = TimelineEnvironment(**env_params)
    agent = make_agent()

    state = environment.reset()
    finished = False
    steps = 0

    print("Starting...")
    print(f"state={state}h (interval={environment.decision_interval}h)")

    # Finite env: stop when finished. Infinite: require max_steps.
    while not finished and (max_steps is None or steps < max_steps):
        if environment.length is None and max_steps is None:
            raise ValueError("max_steps must be set when environment.length is None")
        action, action_values = agent.choose_action(state, environment)
        # earned_reward = environment.payout(action) ## only if tracking payouts
        state, finished = environment.step(action)
        steps += 1

        print(f"Day {steps}: options={action_values}")
        print(f"  Chose {action}")

    print("Done.")


if __name__ == "__main__":
    run()
