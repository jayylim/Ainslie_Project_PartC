/** Port of environments/sticky_timeline.py */

import { TimelineEnvironment } from "./timeline.js";
import { BayesianStickinessBelief } from "./bayesian_stickiness.js";

const OTHER = { dessert: "resist", resist: "dessert" };

export class StickyTimelineEnvironment extends TimelineEnvironment {
  constructor({
    rewards,
    length = null,
    start = 0,
    decision_interval = 24,
    prior_alpha = 1.0,
    prior_beta = 1.0,
    beta_belief = null,
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
    if (this.beta_belief != null) this.beta_belief.reset();
    return this.time_step;
  }

  _opposite(action) {
    if (!(action in OTHER)) throw new Error(`Unknown action ${action}`);
    return OTHER[action];
  }

  step(intent) {
    if (!(intent in this.action_map)) {
      throw new Error(`Unknown intent ${intent}`);
    }

    const realized = intent;
    let stuck = null;
    if (this.last_action != null) {
      stuck = intent === this.last_action ? 1 : 0;
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
      aligned: true,
      s_hat: this.s_hat(),
    };
    return [this.time_step, finished, info];
  }
}
