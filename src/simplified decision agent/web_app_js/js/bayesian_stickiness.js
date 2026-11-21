/** Port of learning/bayesian_stickiness.py */

function lgamma(z) {
  // Lanczos approximation (sufficient for Beta PDF plotting)
  const g = 7;
  const p = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028,
    771.32342877765313, -176.61502916214059, 12.507343278686905,
    -0.13857109526572012, 9.9843696545622154e-6, 1.5056327351493116e-7,
  ];
  if (z < 0.5) {
    return Math.log(Math.PI / Math.sin(Math.PI * z)) - lgamma(1 - z);
  }
  z -= 1;
  let x = p[0];
  for (let i = 1; i < g + 2; i++) x += p[i] / (z + i);
  const t = z + g + 0.5;
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x);
}

export class BayesianStickinessBelief {
  constructor(alpha = 1.0, beta = 1.0) {
    if (alpha <= 0 || beta <= 0) {
      throw new Error("Beta prior parameters must be positive");
    }
    this.alpha0 = Number(alpha);
    this.beta0 = Number(beta);
    this.alpha = Number(alpha);
    this.beta = Number(beta);
  }

  reset() {
    this.alpha = this.alpha0;
    this.beta = this.beta0;
  }

  update(stuck) {
    if (stuck !== 0 && stuck !== 1) {
      throw new Error(`stuck must be 0 or 1, got ${stuck}`);
    }
    if (stuck) this.alpha += 1.0;
    else this.beta += 1.0;
  }

  prior_mean() {
    return this.alpha0 / (this.alpha0 + this.beta0);
  }

  mean() {
    return this.alpha / (this.alpha + this.beta);
  }

  mode() {
    if (this.alpha > 1.0 && this.beta > 1.0) {
      return (this.alpha - 1.0) / (this.alpha + this.beta - 2.0);
    }
    return this.mean();
  }

  n_obs() {
    return Math.round(this.alpha - this.alpha0 + (this.beta - this.beta0));
  }

  n_stick() {
    return Math.round(this.alpha - this.alpha0);
  }

  n_switch() {
    return Math.round(this.beta - this.beta0);
  }

  static _log_beta(a, b) {
    return lgamma(a) + lgamma(b) - lgamma(a + b);
  }

  pdf(x) {
    if (x <= 0.0 || x >= 1.0) {
      if (x === 0.0 && this.alpha < 1.0) return Infinity;
      if (x === 1.0 && this.beta < 1.0) return Infinity;
      return 0.0;
    }
    const log_p =
      (this.alpha - 1.0) * Math.log(x) +
      (this.beta - 1.0) * Math.log(1.0 - x) -
      BayesianStickinessBelief._log_beta(this.alpha, this.beta);
    return Math.exp(log_p);
  }

  pdf_grid(n = 201) {
    if (n < 3) throw new Error("n must be >= 3");
    const xs = [];
    for (let i = 0; i < n; i++) xs.push(i / (n - 1));
    const eps = 1e-6;
    xs[0] = eps;
    xs[xs.length - 1] = 1.0 - eps;
    const dens = xs.map((x) => this.pdf(x));
    return [xs, dens];
  }
}
