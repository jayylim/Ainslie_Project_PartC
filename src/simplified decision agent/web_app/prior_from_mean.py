"""
Map a target stickiness mean s to the smallest positive integer Beta prior.

For prior mean α / (α + β) = s ∈ (0, 1), the lowest positive integers are the
reduced fraction: if s = p / q in lowest terms, then α = p and β = q − p.
"""

from __future__ import annotations

from fractions import Fraction


def integers_for_mean(s, max_denominator=10_000):
    """
    Return (alpha, beta, achieved_mean) for the smallest positive integers
    with alpha / (alpha + beta) equal to the best rational for s.

    Raises ValueError if s is not in (0, 1) or no valid prior exists.
    """
    try:
        s = float(s)
    except (TypeError, ValueError) as exc:
        raise ValueError("stickiness mean must be a number") from exc
    if s != s or s <= 0.0 or s >= 1.0:
        raise ValueError("stickiness mean must be in (0, 1)")

    frac = Fraction(s).limit_denominator(int(max_denominator))
    alpha = int(frac.numerator)
    total = int(frac.denominator)
    beta = total - alpha
    if alpha <= 0 or beta <= 0:
        raise ValueError("could not find positive integer α, β for that mean")
    achieved = alpha / (alpha + beta)
    return alpha, beta, achieved
