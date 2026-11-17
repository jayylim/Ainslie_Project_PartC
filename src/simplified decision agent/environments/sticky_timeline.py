"""
Timeline with a true stickiness process.

Generative rule after the first free choice:
  with probability s_true  -> stick  (realized = last_action)
  otherwise                -> switch (realized = opposite of last_action)

Agent intent is logged but does not override the rolled realization after day 0.
Bayesian belief (if attached) updates from stick/switch bits only; s_true never changes.

simulate() / rewards stay memoryless — foresight geometry is unchanged.
"""

#from __future__ import annotations

import random

from environments.timeline import TimelineEnvironment

OTHER = {"snack": "resist", "resist": "snack"}


class StickyTimelineEnvironment(TimelineEnvironment):
    def __init__(self, rewards, length=None, start=0, decision_interval=24, true_stickiness=0.8, beta_belief=None, seed=None,):
        super().__init__( # using timeline environment initialisations
            rewards=rewards,
            length=length,
            start=start,
            decision_interval=decision_interval,
        )
        if not 0.0 <= true_stickiness <= 1.0:
            raise ValueError("true_stickiness must be in [0, 1]")
        # new parameters for learned stickiness timeline
        self.true_stickiness = float(true_stickiness)
        self.beta_belief = beta_belief  # Beta posterior over s | None; never mutates s_true
        self.seed = seed
        self.rng = random.Random(seed)

        self.last_action = None
        self.stick_history = []  # 1 = stick, 0 = switch (no entry on first choice)
        self.intent_history = []
        self.action_history = []

    def s_hat(self):
        """Posterior mean ŝ = E[s | data] from the attached beta_belief, or None."""
        if self.beta_belief is None:
            return None
        return self.beta_belief.mean()

    def reset(self):
        self.time_step = self.start
        self.last_action = None
        self.stick_history = []
        self.intent_history = []
        self.action_history = []
        self.rng = random.Random(self.seed)
        if self.beta_belief is not None:
            self.beta_belief.reset()
        return self.time_step

    def _opposite(self, action):
        try:
            return OTHER[action]
        except KeyError as exc:
            raise ValueError(f"Unknown action {action!r}") from exc

    def step(self, intent):
        """
        Returns (next_state, finished, info) where info has:
          intent, realized, stuck (None on first choice), s_hat
        """
        if intent not in self.action_map:
            raise ValueError(f"Unknown intent {intent!r}")

        # agent's first choice is always deterministic
        if self.last_action is None:
            realized = intent
            stuck = None
        # stochastic stickiness
        else:
            if self.rng.random() < self.true_stickiness:
                realized = self.last_action
                stuck = 1
            else:
                realized = self._opposite(self.last_action)
                stuck = 0

        self.intent_history.append(intent)
        self.action_history.append(realized)
        if stuck is not None:
            self.stick_history.append(stuck)
            if self.beta_belief is not None:
                self.beta_belief.update(stuck)

        self.last_action = realized
        self.time_step += self.decision_interval

        if self.length is None:
            finished = False
        else:
            finished = self.time_step >= self.length

        info = {
            "intent": intent,
            "realized": realized,
            "stuck": stuck,
            "true_stickiness": self.true_stickiness,
            "s_hat": self.s_hat(),
        }
        return self.time_step, finished, info
