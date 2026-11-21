/** Port of environments/sticky_timeline.py */

import { TimelineEnvironment } from "./timeline.js";
import { BayesianStickinessBelief } from "./bayesian_stickiness.js";

const OTHER = { dessert: "resist", resist: "dessert" };

/** Mulberry32 — deterministic when seed is set; Math.random when seed is null. */
function makeRng(seed) {
  if (seed == null) {
    return () => Math.random();
  }
  let t = seed >>> 0;
  return function random() {
    t += 0x6d2b79f5;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

export class StickyTimelineEnvironment extends TimelineEnvironment {
  constructor({
    rewards,
    length = null,
    start = 0,
    decision_interval = 24,
    prior_alpha = 1.0,
    prior_beta = 1.0,
    beta_belief = null,
    seed = null,
  }) {
    super({ rewards, length, start, decision_interval });
    if (prior_alpha != null && prior_alpha <= 0) {
      throw new Error("prior_alpha must be positive");
    }
    if (prior_beta != null && prior_beta <= 0) {
      throw new Error("prior_beta must be positive");
    }
    this.prior_alpha = Number(prior_alpha != null ? prior_alpha : 1.0);
    this.prior_beta = Number(prior_beta != null ? prior_beta : 1.0);
    this.beta_belief =
      beta_belief != null
        ? beta_belief
        : new BayesianStickinessBelief(this.prior_alpha, this.prior_beta);
    this.seed = seed;
    this.rng = makeRng(seed);
    this.last_action = null;
    this.stick_history = [];
    this.intent_history = [];
    this.action_history = [];
  }

  s_hat() {
    if (this.beta_belief == null) return null;
    return this.beta_belief.mean();
  }

  reset() {
    this.time_step = this.start;
    this.last_action = null;
    this.stick_history = [];
    this.intent_history = [];
    this.action_history = [];
    this.rng = makeRng(this.seed);
    if (this.beta_belief != null) this.beta_belief.reset();
    return this.time_step;
  }

  _opposite(action) {
    if (!(action in OTHER)) throw new Error(`Unknown action ${action}`);
    return OTHER[action];
  }

  step(intent, stickiness) {
    if (!(intent in this.action_map)) {
      throw new Error(`Unknown intent ${intent}`);
    }
    const s = Number(stickiness);
    if (!(s >= 0.0 && s <= 1.0)) {
      throw new Error("stickiness must be in [0, 1]");
    }

    let realized;
    let stuck;
    if (this.last_action == null) {
      realized = intent;
      stuck = null;
    } else if (this.rng() < s) {
      realized = this.last_action;
      stuck = 1;
    } else {
      realized = this._opposite(this.last_action);
      stuck = 0;
    }

    this.intent_history.push(intent);
    this.action_history.push(realized);
    if (stuck != null) {
      this.stick_history.push(stuck);
      if (this.beta_belief != null) this.beta_belief.update(stuck);
    }

    this.last_action = realized;
    this.time_step += this.decision_interval;
    const finished =
      this.length == null ? false : this.time_step >= this.length;

    const info = {
      intent,
      realized,
      stuck,
      aligned: realized === intent,
      stickiness: s,
      s_hat: this.s_hat(),
    };
    return [this.time_step, finished, info];
  }
}
