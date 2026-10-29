"""
Hyperbolic discounter with sticky foresight and a greedy policy.

Same interface as HypAgent. Long vision aggregates expected day values under
stickiness s = P(a_{t+1} = a_t) instead of assuming pure commitment:

    stick_p_t = 1/2 * (1 + (2s - 1)^t)
    V(a) = sum_t [ stick_p_t * V_hyp(a, t) + (1 - stick_p_t) * V_hyp(b, t) ]

When s = 1, long-vision values match HypAgent (pure commitment).
Acts only through the environment interface (simulate / get_actions).
"""


class StickyHypAgent:
    def __init__(
        self, discount_factor=0.8, vision="short", horizon=None, stickiness=1.0
    ):
        self.discount_factor = discount_factor
        self.vision = vision  # "short" or "long"
        self.horizon = horizon  # max # of decisions to plan; None => all remaining if finite
        self.stickiness = stickiness  # s in [0, 1]; P(next action equals current)

    def value_function(self, reward, distance):
        return reward / (1 + self.discount_factor * distance)  # hyperbolic discounting formula

    def get_horizon(self, state, environment):
        """Number of decision nodes counted from `state` (inclusive).
        short vision => always 1
        long vision  => self.horizon, capped by remaining env decisions if finite.
        """
        if self.vision == "short":
            return 1

        if self.horizon == 0:
            raise ValueError("Horizon cannot be 0")

        interval = getattr(environment, "decision_interval", 1)

        # Finite env: never plan past the last real choice node
        if environment.length is not None:
            remaining_hours = max(environment.length - state, 0)
            remaining_decisions = max(
                (remaining_hours + interval - 1) // interval, 1
            )
            if self.horizon is not None:
                return min(self.horizon, remaining_decisions)
            return remaining_decisions

        if self.horizon is not None:
            return self.horizon

        raise ValueError("Horizon must be set as timeline is infinite")

    def stick_p_t(self, t):
        """P(a_t = a | a_0 = a) = 1/2 * (1 + (2s - 1)^t)."""
        s = self.stickiness
        return 0.5 * (1 + (2 * s - 1) ** t)

    def _other_action(self, action, environment, state):
        actions = [a for a in environment.get_actions(state) if a != action]
        if len(actions) != 1:
            raise ValueError(
                "sticky valuation assumes exactly two actions; "
                f"got other={actions!r} for action={action!r}"
            )
        return actions[0]

    def day_value(self, state, action, environment, t):
        """Hyperbolic value of `action`'s outcome delayed by t decision intervals."""
        _, outcomes = environment.simulate(state, action)
        interval = getattr(environment, "decision_interval", 1)
        total = 0.0
        for reward, delay in outcomes.values():
            total += self.value_function(reward, delay + t * interval)
        return total

    # Array for the horizon of sticky expected day values
    def choice_horizon(self, state, action, environment):
        """(t, expected_day_value) pairs over the vision horizon under stickiness.

        expected_day_value = stick_p_t * V_hyp(a, t) + (1 - stick_p_t) * V_hyp(b, t)
        """
        events = []
        horizon = self.get_horizon(state, environment)
        other = self._other_action(action, environment, state)

        for t in range(horizon):
            sticking = self.stick_p_t(t)
            expected = (
                sticking * self.day_value(state, action, environment, t)
                + (1 - sticking) * self.day_value(state, other, environment, t)
            )
            events.append((t, expected))

        return events

    # making a sticky bundled choice across the horizon
    def aggregate_value(self, state, action, environment):
        """Sum expected discounted value of choosing `action` today under stickiness."""
        return sum(
            day_value for _, day_value in self.choice_horizon(state, action, environment)
        )

    def evaluate_action(self, state, action, environment):
        # short vision: evaluate the immediate reward and skip aggregate
        if self.vision == "short":
            _, outcomes = environment.simulate(state, action)
            return max(
                self.value_function(reward, distance)
                for reward, distance in outcomes.values()
            )
        if self.vision == "long":
            return self.aggregate_value(state, action, environment)
        raise ValueError(f"Unknown vision: {self.vision}")

    def choose_action(self, state, environment):
        best_action = None
        best_value = float("-inf")
        action_values = {}

        # loop through all actions and evaluate them
        for action in environment.get_actions(state):
            value = self.evaluate_action(state, action, environment)
            action_values[action] = float(value)
            if value > best_value:
                best_value = value
                best_action = action

        return best_action, action_values
