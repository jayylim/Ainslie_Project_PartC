"""
Timeline with stick/switch rolls driven by perceived stickiness.

Generative rule after the first free choice:
  with probability stickiness  -> stick  (realized = last_action)
  otherwise                    -> switch (realized = opposite of last_action)

`stickiness` is the agent's perceived s (prior mean if not learning, else
current Beta posterior mean). Agent intent is logged but does not override
the rolled realization after day 0.

Bayesian belief updates from stick/switch bits only.
prior_alpha / prior_beta set the Beta prior; fixed foresight stickiness is the
prior mean α/(α+β). Pass beta_belief explicitly to override auto-construction.

simulate() / rewards stay memoryless — foresight geometry is unchanged.
"""

#from __future__ import annotations

import random

from environments.timeline import TimelineEnvironment
from learning.bayesian_stickiness import BayesianStickinessBelief

OTHER = {"snack": "resist", "resist": "snack"}


class StickyTimelineEnvironment(TimelineEnvironment):
    def __init__(self, rewards, length=None, start=0, decision_interval=24, prior_alpha=1.0, prior_beta=1.0, beta_belief=None, seed=None):
        super().__init__( # using timeline environment initialisations
            rewards=rewards,
            length=length,
            start=start,
            decision_interval=decision_interval,
        )
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

    def step(self, intent, stickiness):
        """
        Stick/switch with probability `stickiness` (agent's perceived s).

        Returns (next_state, finished, info) where info has:
          intent, realized, stuck (None on first choice), stickiness, s_hat
        """
        if intent not in self.action_map:
            raise ValueError(f"Unknown intent {intent!r}")

        s = float(stickiness)
        if not 0.0 <= s <= 1.0:
            raise ValueError("stickiness must be in [0, 1]")

        # agent's first choice is always deterministic
        if self.last_action is None:
            realized = intent
            stuck = None
        # stochastic stickiness from perceived s
        else:
            if self.rng.random() < s:
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
            "stickiness": s,
            "s_hat": self.s_hat(),
        }
        return self.time_step, finished, info
