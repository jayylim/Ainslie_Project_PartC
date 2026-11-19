"""
Memoryless sequential timeline: the same SS vs LL offer at every decision node.
Past choices are not stored; delays are fixed relative to the current decision.

Timescale: 1 time_step = 1 hour. Choices every decision_interval hours (default 24).
Reward delays are in hours relative to the choice (e.g. LL delay=12 = next morning).
"""


class TimelineEnvironment:
    def __init__(self, rewards, length=None, start=0, decision_interval=24):
        self.rewards = rewards
        self.length = length  # None => infinite; units = hours
        self.start = start
        self.time_step = start
        self.decision_interval = decision_interval  # hours between choices
        self.action_map = {
            "snack": "SS",
            "resist": "LL",
        }

    # === Agent-facing layer ===
    def get_actions(self, state=None):
        return ["snack", "resist"]

    def simulate(self, state, action):
        """One-step lookahead without mutating environment time."""
        next_state = state + self.decision_interval
        reward_type, reward_info = self.get_reward(action)
        outcomes = {
            reward_type: (reward_info["value"], reward_info["delay"])
        }
        return next_state, outcomes

    # === Environment methods ===
    def reset(self):
        self.time_step = self.start
        return self.time_step

    def get_reward(self, action):
        reward_type = self.action_map[action]
        return reward_type, self.rewards[reward_type]

    # Advance the time step of environment
    def step(self, action):
        self.time_step += self.decision_interval

        if self.length is None:
            finished = False
        else:
            finished = self.time_step >= self.length

        return self.time_step, finished

    # for tracking reward payouts (only matters for paradigms with memory)
    def payout(self, action):
        _, reward_info = self.get_reward(action)
        # Credit only if immediate (delay == 0). Delayed offers are planning-only.
        if reward_info["delay"] == 0:
            return reward_info["value"]
        return 0
