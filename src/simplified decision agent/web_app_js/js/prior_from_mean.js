/**
 * Port of web_app/prior_from_mean.py
 *
 * Map stickiness mean s ∈ (0, 1) to the smallest positive integer Beta prior
 * via best rational approximation (continued-fraction limit_denominator).
 */

function limitDenominator(x, maxDenominator = 10000) {
  const maxD = Math.max(1, Math.trunc(maxDenominator));
  if (!Number.isFinite(x)) throw new Error("stickiness mean must be a number");

  // Continued-fraction convergent search (Python Fraction.limit_denominator).
  let a0 = Math.floor(x);
  let n0 = 1;
  let d0 = 0;
  let n1 = a0;
  let d1 = 1;
  let frac = x - a0;
  if (d1 > maxD) return [a0, 1];

  while (frac > 1e-18) {
    frac = 1.0 / frac;
    const a = Math.floor(frac);
    const n2 = a * n1 + n0;
    const d2 = a * d1 + d0;
    if (d2 > maxD) {
      // Bound last semi-convergent so denominator ≤ maxD.
      const k = Math.floor((maxD - d0) / d1);
      const nBound = k * n1 + n0;
      const dBound = k * d1 + d0;
      const errBound = Math.abs(x - nBound / dBound);
      const errPrev = Math.abs(x - n1 / d1);
      return errBound < errPrev ? [nBound, dBound] : [n1, d1];
    }
    n0 = n1;
    d0 = d1;
    n1 = n2;
    d1 = d2;
    frac -= a;
    if (Math.abs(x - n1 / d1) < 1e-15) break;
  }
  return [n1, d1];
}

export function integers_for_mean(s, max_denominator = 10000) {
  const x = Number(s);
  if (!Number.isFinite(x) || x <= 0.0 || x >= 1.0) {
    throw new Error("stickiness mean must be in (0, 1)");
  }
  const [alpha, total] = limitDenominator(x, max_denominator);
  const beta = total - alpha;
  if (alpha <= 0 || beta <= 0) {
    throw new Error("could not find positive integer α, β for that mean");
  }
  const achieved = alpha / (alpha + beta);
  return [alpha, beta, achieved];
}
