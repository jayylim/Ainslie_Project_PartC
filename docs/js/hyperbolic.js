/** Port of agents/hyperbolic.py */

export class HypAgent {
  constructor({ discount_factor = 0.8, vision = "short", horizon = null } = {}) {
    this.discount_factor = discount_factor;
    this.vision = vision;
    this.horizon = horizon;
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

  choice_horizon(state, action, environment) {
    const events = [];
    let current_state = state;
    const horizon = this.get_horizon(state, environment);
    const interval = environment.decision_interval ?? 1;

    for (let t = 0; t < horizon; t++) {
      const [next_state, outcomes] = environment.simulate(current_state, action);
      for (const [reward, delay] of Object.values(outcomes)) {
        events.push([delay + t * interval, reward]);
      }
      current_state = next_state;
    }
    return events;
  }

  aggregate_value(state, action, environment) {
    let total = 0;
    for (const [delay, reward] of this.choice_horizon(state, action, environment)) {
      total += this.value_function(reward, delay);
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
