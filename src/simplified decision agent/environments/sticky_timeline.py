"""
Timeline where the agent's intent is always realized.

After the first free choice, each step observes whether the agent stuck or
switched relative to the previous realized action:

  stuck = 1  if intent == last_action
  stuck = 0  otherwise

The Beta belief updates from those stick/switch bits only. There is no
stochastic stickiness roll — perceived stickiness affects foresight valuation
in StickyHypAgent when learn_stickiness is On, not the environment outcome.

prior_alpha / prior_beta set the Beta prior; fixed foresight stickiness is the
prior mean α/(α+β). Pass beta_belief explicitly to override auto-construction.

simulate() / rewards stay memoryless — foresight geometry is unchanged.
"""

from environments.timeline import TimelineEnvironment
from learning.bayesian_stickiness import BayesianStickinessBelief

OTHER = {"dessert": "resist", "resist": "dessert"}


class StickyTimelineEnvironment(TimelineEnvironment):
    def __init__(
        self,
        rewards,
        length=None,
        start=0,
        decision_interval=24,
        prior_alpha=1.0,
        prior_beta=1.0,
        beta_belief=None,
        seed=None,
    ):
        super().__init__(
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
        # seed retained for API compatibility; outcomes are deterministic
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

    def step(self, intent):
        """
        Realize `intent` and update stickiness belief from the agent's choice.

        Returns (next_state, finished, info) where info has:
          intent, realized (= intent), stuck (None on first choice), s_hat
        """
        if intent not in self.action_map:
            raise ValueError(f"Unknown intent {intent!r}")

        realized = intent
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
            "aligned": True,
            "s_hat": self.s_hat(),
        }
        return self.time_step, finished, info
