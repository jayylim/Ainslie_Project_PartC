/** Port of agents/sticky_agent.py */

export class StickyHypAgent {
  constructor({
    discount_factor = 0.8,
    vision = "short",
    horizon = null,
    perceived_s = 1.0,
    learn_stickiness = false,
  } = {}) {
    this.discount_factor = discount_factor;
    this.vision = vision;
    this.horizon = horizon;
    this.perceived_s = Number(perceived_s);
    this.learn_stickiness = Boolean(learn_stickiness);
  }

  perceived_stickiness(environment) {
    if (!this.learn_stickiness) return this.perceived_s;
    const beta_belief = environment ? environment.beta_belief : null;
    if (beta_belief == null) {
      throw new Error("learn_stickiness requires environment.beta_belief");
    }
    return beta_belief.mean();
  }

  value_function(reward, distance) {
    return reward / (1 + this.discount_factor * distance);
  }

  get_horizon(state, environment) {
    if (this.vision === "short") return 1;
    if (this.horizon === 0) throw new Error("Horizon cannot be 0");

    const interval = environment.decision_interval ?? 1;

    if (environment.length != null) {
      const remaining_hours = Math.max(environment.length - state, 0);
      const remaining_decisions = Math.max(
        Math.floor((remaining_hours + interval - 1) / interval),
        1
      );
      if (this.horizon != null) return Math.min(this.horizon, remaining_decisions);
      return remaining_decisions;
    }

    if (this.horizon != null) return this.horizon;
    throw new Error("Horizon must be set as timeline is infinite");
  }

  stick_p_t(t, environment = null) {
    const s = this.perceived_stickiness(environment);
    return 0.5 * (1 + (2 * s - 1) ** t);
  }

  _other_action(action, environment, state) {
    const actions = environment.get_actions(state).filter((a) => a !== action);
    if (actions.length !== 1) {
      throw new Error(
        `sticky valuation assumes exactly two actions; got other=${JSON.stringify(actions)} for action=${action}`
      );
    }
    return actions[0];
  }

  day_value(state, action, environment, t) {
    const [, outcomes] = environment.simulate(state, action);
    const interval = environment.decision_interval ?? 1;
    let total = 0.0;
    for (const [reward, delay] of Object.values(outcomes)) {
      total += this.value_function(reward, delay + t * interval);
    }
    return total;
  }

  choice_horizon(state, action, environment) {
    const events = [];
    const horizon = this.get_horizon(state, environment);
    const other = this._other_action(action, environment, state);

    for (let t = 0; t < horizon; t++) {
      const sticking = this.stick_p_t(t, environment);
      const expected =
        sticking * this.day_value(state, action, environment, t) +
        (1 - sticking) * this.day_value(state, other, environment, t);
      events.push([t, expected]);
    }
    return events;
  }

  aggregate_value(state, action, environment) {
    let total = 0;
    for (const [, day_value] of this.choice_horizon(state, action, environment)) {
      total += day_value;
    }
    return total;
  }

  evaluate_action(state, action, environment) {
    if (this.vision === "short") {
      const [, outcomes] = environment.simulate(state, action);
      let best = -Infinity;
      for (const [reward, distance] of Object.values(outcomes)) {
        best = Math.max(best, this.value_function(reward, distance));
      }
      return best;
    }
    if (this.vision === "long") {
      return this.aggregate_value(state, action, environment);
    }
    throw new Error(`Unknown vision: ${this.vision}`);
  }

  choose_action(state, environment) {
    let best_action = null;
    let best_value = -Infinity;
    const action_values = {};
    for (const action of environment.get_actions(state)) {
      const value = this.evaluate_action(state, action, environment);
      action_values[action] = Number(value);
      if (value > best_value) {
        best_value = value;
        best_action = action;
      }
    }
    return [best_action, action_values];
  }
}
