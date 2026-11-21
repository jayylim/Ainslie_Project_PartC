/** Port of environments/timeline.py */

export class TimelineEnvironment {
  constructor({ rewards, length = null, start = 0, decision_interval = 24 }) {
    this.rewards = rewards;
    this.length = length;
    this.start = start;
    this.time_step = start;
    this.decision_interval = decision_interval;
    this.action_map = {
      dessert: "SS",
      resist: "LL",
    };
  }

  get_actions(_state = null) {
    return ["dessert", "resist"];
  }

  simulate(state, action) {
    const next_state = state + this.decision_interval;
    const [reward_type, reward_info] = this.get_reward(action);
    const outcomes = {
      [reward_type]: [reward_info.value, reward_info.delay],
    };
    return [next_state, outcomes];
  }

  reset() {
    this.time_step = this.start;
    return this.time_step;
  }

  get_reward(action) {
    const reward_type = this.action_map[action];
    return [reward_type, this.rewards[reward_type]];
  }

  step(action) {
    this.time_step += this.decision_interval;
    const finished =
      this.length == null ? false : this.time_step >= this.length;
    return [this.time_step, finished];
  }

  payout(action) {
    const [, reward_info] = this.get_reward(action);
    if (reward_info.delay === 0) return reward_info.value;
    return 0;
  }
}
