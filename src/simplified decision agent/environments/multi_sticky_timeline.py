"""
Multi-decision sticky timeline: parallel independent SS/LL choices, shared Beta.

Each decision has its own SS/LL magnitudes and delays (SS delay must be 0).
One environment step advances decision_interval hours (default 24) and realizes
an intent for EVERY decision in that period — all faced at the same clock time.

Stick/switch is tracked per decision against that decision's last action.
All stick/switch observations update one shared BayesianStickinessBelief.

Intent is always realized. Perceived stickiness affects foresight in
StickyHypAgent when learn_stickiness is On, not the environment outcome.
"""

from __future__ import annotations

from learning.bayesian_stickiness import BayesianStickinessBelief

ACTION_MAP = {
    "cheat": "SS",
    "resist": "LL",
}


class _DecisionView:
    """Single-decision adapter so StickyHypAgent valuation runs unchanged."""

    def __init__(self, parent: "MultiStickyTimelineEnvironment", decision_id: str):
        self._parent = parent
        self.decision_id = decision_id
        self.length = parent.length
        self.start = parent.start
        self.decision_interval = parent.decision_interval
        self.beta_belief = parent.beta_belief
        self.action_map = ACTION_MAP
        self.rewards = parent.decisions[decision_id]

    @property
    def time_step(self):
        return self._parent.time_step

    def get_actions(self, state=None):
        return ["cheat", "resist"]

    def simulate(self, state, action):
        return self._parent.simulate(state, action, self.decision_id)

    def s_hat(self):
        return self._parent.s_hat()


class MultiStickyTimelineEnvironment:
    def __init__(
        self,
        decisions,
        length=None,
        start=0,
        decision_interval=24,
        prior_alpha=1.0,
        prior_beta=1.0,
        beta_belief=None,
    ):
        if not decisions:
            raise ValueError("decisions must be a non-empty mapping")
        if decision_interval is None or decision_interval <= 0:
            raise ValueError("decision_interval must be a positive number of hours")

        self.decisions = {}
        self.decision_ids = []
        for decision_id, spec in decisions.items():
            if "SS" not in spec or "LL" not in spec:
                raise ValueError(
                    f"decision {decision_id!r} must have SS and LL reward entries"
                )
            ss = spec["SS"]
            if ss.get("delay", 0) != 0:
                raise ValueError(
                    f"decision {decision_id!r}: SS delay must be 0, got {ss.get('delay')!r}"
                )
            self.decision_ids.append(decision_id)
            self.decisions[decision_id] = {
                "SS": dict(ss),
                "LL": dict(spec["LL"]),
            }

        self.length = length
        self.start = start
        self.time_step = start
        self.decision_interval = decision_interval
        self.action_map = ACTION_MAP

        if prior_alpha is not None and prior_alpha <= 0:
            raise ValueError("prior_alpha must be positive")
        if prior_beta is not None and prior_beta <= 0:
            raise ValueError("prior_beta must be positive")
        self.prior_alpha = float(prior_alpha if prior_alpha is not None else 1.0)
        self.prior_beta = float(prior_beta if prior_beta is not None else 1.0)
        if beta_belief is not None:
            self.beta_belief = beta_belief
        else:
            self.beta_belief = BayesianStickinessBelief(
                alpha=self.prior_alpha, beta=self.prior_beta
            )

        self.last_action = {did: None for did in self.decision_ids}
        # (decision_id, stuck) for debugging; stuck only when not first choice
        self.stick_history = []
        self.intent_history = []  # list of intent dicts per period
        self.action_history = []  # list of realized dicts per period

    def s_hat(self):
        """Posterior mean ŝ = E[s | data] from the shared beta_belief, or None."""
        if self.beta_belief is None:
            return None
        return self.beta_belief.mean()

    def decision_view(self, decision_id):
        if decision_id not in self.decisions:
            raise ValueError(f"Unknown decision_id {decision_id!r}")
        return _DecisionView(self, decision_id)

    def get_actions(self, state=None, decision_id=None):
        if decision_id is None:
            raise ValueError("decision_id is required")
        if decision_id not in self.decisions:
            raise ValueError(f"Unknown decision_id {decision_id!r}")
        return ["cheat", "resist"]

    def get_reward(self, action, decision_id):
        if decision_id not in self.decisions:
            raise ValueError(f"Unknown decision_id {decision_id!r}")
        if action not in ACTION_MAP:
            raise ValueError(f"Unknown action {action!r}")
        reward_type = ACTION_MAP[action]
        return reward_type, self.decisions[decision_id][reward_type]

    def simulate(self, state, action, decision_id):
        """One-step lookahead for one decision without mutating environment time.

        Delays are relative to the shared current clock (all decisions at once).
        Foresight repeats are spaced by decision_interval.
        """
        next_state = state + self.decision_interval
        reward_type, reward_info = self.get_reward(action, decision_id)
        outcomes = {
            reward_type: (reward_info["value"], reward_info["delay"])
        }
        return next_state, outcomes

    def reset(self):
        self.time_step = self.start
        self.last_action = {did: None for did in self.decision_ids}
        self.stick_history = []
        self.intent_history = []
        self.action_history = []
        if self.beta_belief is not None:
            self.beta_belief.reset()
        return self.time_step

    def step(self, intents):
        """
        Realize intents for EVERY decision at the current clock time, then
        advance by decision_interval hours.

        `intents` must map every decision_id to "cheat" or "resist".

        Returns (next_state, finished, info) where info has:
          per_decision: {id: {intent, realized, stuck, aligned}}
          s_hat
          period_hours
        """
        if not isinstance(intents, dict):
            raise TypeError(f"intents must be a dict, got {type(intents).__name__}")
        missing = [did for did in self.decision_ids if did not in intents]
        if missing:
            raise ValueError(f"Missing intents for decisions: {missing}")
        extra = [did for did in intents if did not in self.decisions]
        if extra:
            raise ValueError(f"Unknown decision ids in intents: {extra}")

        per_decision = {}
        realized_period = {}
        for decision_id in self.decision_ids:
            intent = intents[decision_id]
            if intent not in ACTION_MAP:
                raise ValueError(
                    f"Unknown intent {intent!r} for decision {decision_id!r}"
                )

            realized = intent
            last = self.last_action[decision_id]
            if last is None:
                stuck = None
            else:
                stuck = 1 if intent == last else 0

            if stuck is not None:
                self.stick_history.append((decision_id, stuck))
                if self.beta_belief is not None:
                    self.beta_belief.update(stuck)

            self.last_action[decision_id] = realized
            realized_period[decision_id] = realized
            per_decision[decision_id] = {
                "intent": intent,
                "realized": realized,
                "stuck": stuck,
                "aligned": True,
            }

        self.intent_history.append(dict(intents))
        self.action_history.append(realized_period)

        self.time_step += self.decision_interval
        if self.length is None:
            finished = False
        else:
            finished = self.time_step >= self.length

        info = {
            "per_decision": per_decision,
            "s_hat": self.s_hat(),
            "period_hours": self.decision_interval,
        }
        return self.time_step, finished, info
