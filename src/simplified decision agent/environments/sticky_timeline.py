"""
Timeline where realized actions always follow agent intent.

Generative rule:
  realized = intent  (every day, including day 0)

After the first free choice, stick/switch is observed from consecutive intents:
  stuck = 1  if intent == last_action
  stuck = 0  if intent != last_action

No RNG: stuck is never sampled; it is only that equality check.

Bayesian belief updates from those stick/switch bits only.

`stickiness` on step() is the agent's foresight s (prior mean if not learning,
else current Beta posterior mean). It is logged for diagnostics; it does not
override realization.

simulate() / rewards stay memoryless — foresight geometry is unchanged.
"""

#from __future__ import annotations

from environments.timeline import TimelineEnvironment
from learning.bayesian_stickiness import BayesianStickinessBelief

OTHER = {"dessert": "resist", "resist": "dessert"}


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
        # Kept for call-site compatibility; unused (no stick/switch sampling).
        self.seed = seed

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
        Realize intent; record stick/switch vs last action for belief updates.

        stuck is deterministic:
          None on first choice; else 1 iff intent == last_action, else 0.
        Never sampled from stickiness.

        Returns (next_state, finished, info) where info has:
          intent, realized (=intent), stuck, stickiness, s_hat
        """
        if intent not in self.action_map:
            raise ValueError(f"Unknown intent {intent!r}")

        s = float(stickiness)
        if not 0.0 <= s <= 1.0:
            raise ValueError("stickiness must be in [0, 1]")

        # Actions always follow the agent's intent.
        realized = intent

        # First choice has no previous action → free; later days observe stick/switch.
        if self.last_action is None:
            stuck = None
        else:
            stuck = 1 if intent == self.last_action else 0

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
            "aligned": True,  # realized always equals intent
            "stickiness": s,
            "s_hat": self.s_hat(),
        }
        return self.time_step, finished, info
