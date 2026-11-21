/** Defaults mirrored from agents/agent_configs.py + main.py */

export const agentParams = {
  discount_factor: 0.6,
  vision: "long",
  horizon: 10,
};

export const stickyParams = {
  discount_factor: 0.6,
  vision: "long",
  horizon: 10,
  perceived_s: 0.6,
  learn_stickiness: true,
};

export const envParams = {
  length: null,
  start: 0,
  decision_interval: 24,
  rewards: {
    SS: { value: 3, delay: 0 },
    LL: { value: 20, delay: 12 },
  },
};

export const stickyEnvParams = {
  prior_alpha: 1,
  prior_beta: 1,
};

export const maxSteps = null;

export const DEFAULT_DELAY_MAX = 24;
export const DEFAULT_DELAY_STEP = 4;
export const DEFAULT_STICKINESS = stickyParams.perceived_s;
export const DEFAULT_LEARN_STICKINESS = stickyParams.learn_stickiness;
export const DEFAULT_PRIOR_ALPHA = Number(stickyEnvParams.prior_alpha);
export const DEFAULT_PRIOR_BETA = Number(stickyEnvParams.prior_beta);
export const DEFAULT_PRIOR_MEAN =
  DEFAULT_PRIOR_ALPHA / (DEFAULT_PRIOR_ALPHA + DEFAULT_PRIOR_BETA);
export const DEFAULT_STICKY_STEP_SEED = null;

export const DEFAULT_CONFIG = {
  ss_value: envParams.rewards.SS.value,
  ss_delay: envParams.rewards.SS.delay,
  ll_value: envParams.rewards.LL.value,
  ll_delay: envParams.rewards.LL.delay,
  length: envParams.length,
  horizon: agentParams.horizon,
  vision: agentParams.vision,
  discount_factor: agentParams.discount_factor,
  max_steps: maxSteps,
};
