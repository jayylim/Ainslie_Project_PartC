/**
 * Port of scan_sticky_learning.py — sticky learning regime scan over prior mean s.
 *
 * Same algorithm as the Flask / sticky_step demo:
 *   StickyTimelineEnvironment + StickyHypAgent(learn_stickiness=true)
 *   prior α, β = lowest positive integers with mean s
 *   each step: greedy intent under ŝ → realize → Beta update from stick/switch
 */

import { integers_for_mean } from "./prior_from_mean.js";
import { StickyHypAgent } from "./sticky_agent.js";
import { StickyTimelineEnvironment } from "./sticky_timeline.js";

export const DEFAULT_SCAN_OPTS = {
  n_steps: 60,
  s_min: 0.5,
  s_max: 0.99,
  s_step: 0.01,
  strength_min: 2,
  strength_max: 200,
};

export function prior_strength(row) {
  return Math.trunc(row.alpha) + Math.trunc(row.beta);
}

export function filter_rows_by_strength(rows, strength_min = null, strength_max = null) {
  let lo = strength_min == null ? 0 : Math.trunc(strength_min);
  let hi = strength_max == null ? 1e9 : Math.trunc(strength_max);
  if (lo > hi) {
    const t = lo;
    lo = hi;
    hi = t;
  }
  return rows.filter((r) => {
    const n = prior_strength(r);
    return lo <= n && n <= hi;
  });
}

export function simulate(s, { n_steps, agent_kwargs, env_kwargs }) {
  const [alpha, beta, achieved] = integers_for_mean(s);
  const environment = new StickyTimelineEnvironment({
    ...env_kwargs,
    prior_alpha: alpha,
    prior_beta: beta,
  });
  const agent = new StickyHypAgent({
    ...agent_kwargs,
    perceived_s: achieved,
    learn_stickiness: true,
  });
  let state = environment.reset();
  const traj = {
    s0: achieved,
    alpha0: alpha,
    beta0: beta,
    actions: [],
    means: [environment.s_hat()],
    values: [],
    stuck: [],
  };
  for (let i = 0; i < n_steps; i++) {
    const [intent, action_values] = agent.choose_action(state, environment);
    traj.values.push([
      Number(action_values.dessert),
      Number(action_values.resist),
    ]);
    const [nextState, _finished, info] = environment.step(intent);
    state = nextState;
    traj.actions.push(intent);
    traj.means.push(environment.s_hat());
    traj.stuck.push(info.stuck);
  }
  return traj;
}

function flipIndices(actions) {
  const flips = [];
  for (let i = 1; i < actions.length; i++) {
    if (actions[i] !== actions[i - 1]) flips.push(i);
  }
  return flips;
}

function runLengths(actions) {
  if (!actions.length) return [];
  const runs = [];
  let cur = actions[0];
  let n = 1;
  for (let i = 1; i < actions.length; i++) {
    if (actions[i] === cur) n += 1;
    else {
      runs.push([cur, n]);
      cur = actions[i];
      n = 1;
    }
  }
  runs.push([cur, n]);
  return runs;
}

function meansNondecreasing(means, eps = 1e-15) {
  for (let i = 0; i < means.length - 1; i++) {
    if (means[i + 1] + eps < means[i]) return false;
  }
  return true;
}

function loopSubtype(actions, flips) {
  if (actions.length < 4 || flips.length < 2) return "sparse_flips";
  const burn = Math.min(5, Math.floor(actions.length / 5));
  const late = actions.slice(burn);
  const runs = runLengths(late);
  if (!runs.length) return "sparse_flips";

  const lengths = runs.map(([, n]) => n);
  const counts = new Map();
  for (const [a] of runs) counts.set(a, (counts.get(a) || 0) + 1);
  let majority = runs[0][0];
  let best = 0;
  for (const [a, c] of counts) {
    if (c > best) {
      best = c;
      majority = a;
    }
  }
  const minorityRuns = runs.filter(([a]) => a !== majority).map(([, n]) => n);
  const majorityRuns = runs.filter(([a]) => a === majority).map(([, n]) => n);

  if (lengths.every((n) => n === 1) && runs.length >= 4) {
    return "period_2_alternation";
  }
  if (
    minorityRuns.length &&
    majorityRuns.length &&
    Math.max(...minorityRuns) <= 2 &&
    Math.min(...majorityRuns) >= 3 &&
    minorityRuns.filter((n) => n === 1).length >=
      Math.max(1, Math.floor(minorityRuns.length / 2))
  ) {
    return `intermittent_${majority}_with_spikes`;
  }
  const gaps = [];
  for (let i = 1; i < flips.length; i++) gaps.push(flips[i] - flips[i - 1]);
  if (
    gaps.length &&
    Math.max(...gaps) - Math.min(...gaps) <= 2 &&
    Math.min(...gaps) >= 2
  ) {
    const meanGap = gaps.reduce((a, b) => a + b, 0) / gaps.length;
    return `quasi_periodic_period_${Math.round(meanGap)}`;
  }
  return "irregular_switching";
}

export function classify(traj) {
  const actions = traj.actions;
  const means = traj.means;
  const flips = flipIndices(actions);
  const rising = meansNondecreasing(means);
  const first = actions.length ? actions[0] : null;
  const last = actions.length ? actions[actions.length - 1] : null;
  const base = {
    n_flips: flips.length,
    first_action: first,
    last_action: last,
    s_start: means[0],
    s_end: means[means.length - 1],
    means_rising: rising,
    flip_days: flips.map((i) => i + 1),
  };

  if (!actions.length) return { label: "empty", detail: "", ...base };
  if (flips.length === 0) {
    return {
      label: rising ? "lock_in_rising_s" : "lock_in",
      detail: `always ${first}`,
      ...base,
    };
  }
  if (flips.length === 1) {
    return {
      label: "one_shot_switch",
      detail: `${first} → ${last} at day ${flips[0] + 1}`,
      ...base,
    };
  }
  return {
    label: "switching_loop",
    detail: loopSubtype(actions, flips),
    ...base,
  };
}

export function contiguous_where(rows, predicate, key = null) {
  const keyFn = key || (() => true);
  const out = [];
  let i = 0;
  while (i < rows.length) {
    if (!predicate(rows[i])) {
      i += 1;
      continue;
    }
    let j = i + 1;
    const k0 = keyFn(rows[i]);
    while (j < rows.length && predicate(rows[j]) && keyFn(rows[j]) === k0) {
      j += 1;
    }
    const chunk = rows.slice(i, j);
    out.push([chunk[0].s, chunk[chunk.length - 1].s, chunk]);
    i = j;
  }
  return out;
}

export function action_string(actions, max_len = 72) {
  let s = actions.map((a) => (a === "dessert" ? "D" : "R")).join("");
  if (s.length <= max_len) return s;
  return s.slice(0, max_len - 1) + "…";
}

export function unique_s_targets(s_min, s_max, s_step) {
  if (!(s_step > 0)) throw new Error("s_step must be positive");
  if (!(0.0 < s_min && s_min < s_max && s_max < 1.0)) {
    throw new Error("need 0 < s_min < s_max < 1");
  }
  const n = Math.round((s_max - s_min) / s_step);
  const s_values = [];
  for (let i = 0; i <= n; i++) {
    const s = s_min + i * s_step;
    if (s > 0.0 && s < 1.0) s_values.push(s);
  }
  const seen = new Set();
  const unique = [];
  for (const s of s_values) {
    const [alpha, beta, achieved] = integers_for_mean(s);
    const key = `${alpha},${beta}`;
    if (seen.has(key)) continue;
    seen.add(key);
    unique.push(achieved);
  }
  return unique;
}

export function build_scan(s_values, { n_steps, agent_kwargs, env_kwargs }) {
  const rows = [];
  for (const s of s_values) {
    const traj = simulate(s, { n_steps, agent_kwargs, env_kwargs });
    const cls = classify(traj);
    rows.push({
      s: traj.s0,
      target_s: Number(s),
      alpha: traj.alpha0,
      beta: traj.beta0,
      n: traj.alpha0 + traj.beta0,
      label: cls.label,
      detail: cls.detail,
      n_flips: cls.n_flips,
      means_rising: cls.means_rising,
      s_end: cls.s_end,
      first: cls.first_action,
      last: cls.last_action,
      flip_days: cls.flip_days,
      actions: traj.actions,
      means: traj.means,
    });
  }
  return rows;
}

export function q1_q2_summary(rows, { n_steps }) {
  if (!rows.length) {
    return {
      text: `steps=${n_steps}  points=0\n\n(no rows)`,
      q1_ranges: [],
      one_shot: [],
    };
  }
  const lines = [
    `steps=${n_steps}  points=${rows.length}  ` +
      `s in [${rows[0].s.toFixed(4)}, ${rows[rows.length - 1].s.toFixed(4)}]`,
    "",
    "Q1 — s continually increases (preference is locked-in / all sticks)",
    `  (Finite horizon: some mid-s locks (especailly to SS) may only mean no switches yet within ${n_steps} steps.)`,
  ];
  const q1 = contiguous_where(
    rows,
    (r) => r.label === "lock_in_rising_s",
    (r) => r.first
  );
  if (!q1.length) lines.push("  (none in this grid)");
  else {
    for (const [s_lo, s_hi, chunk] of q1) {
      const sEnds = chunk.map((c) => c.s_end);
      lines.push(
        `  s ∈ [${s_lo.toFixed(4)}, ${s_hi.toFixed(4)}]  always ${chunk[0].first}  ` +
          `ŝ_end in [${Math.min(...sEnds).toFixed(3)}, ${Math.max(...sEnds).toFixed(3)}]`
      );
    }
  }

  lines.push("", "Q2 — one preference switch, never switches back");
  const one_shot = rows.filter((r) => r.label === "one_shot_switch");
  if (!one_shot.length) lines.push("  (none in this grid / horizon)");
  else {
    for (const r of one_shot) {
      lines.push(
        `  s=${r.s.toFixed(4)}  α=${r.alpha} β=${r.beta}  ` +
          `${r.detail}  ŝ_end=${r.s_end.toFixed(3)}`
      );
      lines.push(`    ${action_string(r.actions)}`);
    }
  }

  return {
    text: lines.join("\n"),
    q1_ranges: q1.map(([s_lo, s_hi, chunk]) => ({
      s_lo,
      s_hi,
      first: chunk[0].first,
      s_end_min: Math.min(...chunk.map((c) => c.s_end)),
      s_end_max: Math.max(...chunk.map((c) => c.s_end)),
    })),
    one_shot: one_shot.map((r) => ({
      s: r.s,
      alpha: r.alpha,
      beta: r.beta,
      detail: r.detail,
      s_end: r.s_end,
    })),
  };
}
