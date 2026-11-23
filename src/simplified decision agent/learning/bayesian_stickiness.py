"""
Beta–Bernoulli belief over stickiness s = P(a_{t+1} = a_t).

Prior:    s ~ Beta(alpha0, beta0)
Observe:  stuck ∈ {0, 1}  (1 = stick, 0 = switch)
          from whether consecutive followed-through intents match
Posterior: Beta(alpha0 + #sticks, beta0 + #switches)

Perceived stickiness for foresight is the prior mean α₀/(α₀+β₀) when not
learning, else the posterior mean α/(α+β).
"""

from __future__ import annotations

import math


class BayesianStickinessBelief:
    def __init__(self, alpha=1.0, beta=1.0):
        if alpha <= 0 or beta <= 0:
            raise ValueError("Beta prior parameters must be positive")
        self.alpha0 = float(alpha)
        self.beta0 = float(beta)
        self.alpha = float(alpha)
        self.beta = float(beta)

    def reset(self):
        """Restore the prior."""
        self.alpha = self.alpha0
        self.beta = self.beta0

    def update(self, stuck):
        """Bayesian update from one stick (1) / switch (0) observation."""
        if stuck not in (0, 1):
            raise ValueError(f"stuck must be 0 or 1, got {stuck!r}")
        if stuck:
            self.alpha += 1.0
        else:
            self.beta += 1.0

    def prior_mean(self):
        """Fixed prior stickiness α₀ / (α₀ + β₀) before any stick/switch obs."""
        return self.alpha0 / (self.alpha0 + self.beta0)

    def mean(self):
        # computes the posterior mean E[s | data] = alpha / (alpha + beta)
        return self.alpha / (self.alpha + self.beta)

    def mode(self):
        # computes the posterior mode when alpha, beta > 1; else falls back to mean
        if self.alpha > 1.0 and self.beta > 1.0:
            return (self.alpha - 1.0) / (self.alpha + self.beta - 2.0)
        return self.mean()

    def n_obs(self):
        # counts the total number of stick/switch observations since the prior
        return int(round((self.alpha - self.alpha0) + (self.beta - self.beta0)))

    def n_stick(self):
        return int(round(self.alpha - self.alpha0))

    def n_switch(self):
        return int(round(self.beta - self.beta0))

    @staticmethod
    def _log_beta(a, b):
        return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)

    # probability density function of the beta distribution
    def pdf(self, x):
        if x <= 0.0 or x >= 1.0:
            # Endpoints: infinite if param < 1, finite/zero otherwise — treat as 0 for plotting
            if x == 0.0 and self.alpha < 1.0:
                return float("inf")
            if x == 1.0 and self.beta < 1.0:
                return float("inf")
            return 0.0
        # p(x) = x^{a-1} (1-x)^{b-1} / B(a, b)
        log_p = (
            (self.alpha - 1.0) * math.log(x)
            + (self.beta - 1.0) * math.log(1.0 - x)
            - self._log_beta(self.alpha, self.beta)
        )
        return math.exp(log_p)

    def pdf_grid(self, n=201):
        """Return (xs, dens) for plotting on (0, 1), endpoints excluded."""
        if n < 3:
            raise ValueError("n must be >= 3")
        xs = [i / (n - 1) for i in range(n)]
        # Avoid exact 0/1 for stable logs when alpha or beta < 1
        eps = 1e-6
        xs[0] = eps
        xs[-1] = 1.0 - eps
        dens = [self.pdf(x) for x in xs]
        return xs, dens
