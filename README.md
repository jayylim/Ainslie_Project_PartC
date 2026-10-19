# Simplified Decision Agent

Hyperbolic discounter on a memoryless SS/LL timeline (Ainslie-style bundling). Edit params in `main.py`, then run from this directory.

## Architecture canvas

Interactive outline (read order, runtime, visual demos) — open in Cursor beside the chat:

**[Architecture Outline](/Users/joshualim/.cursor/projects/Users-joshualim-Desktop-Part-C-Ainslie-Agents/canvases/architecture-outline.canvas.tsx)**

```
/Users/joshualim/.cursor/projects/Users-joshualim-Desktop-Part-C-Ainslie-Agents/canvases/architecture-outline.canvas.tsx
```

That path is the Cursor-managed canvas for this workspace (not inside the git tree). In Cursor, click the link above or open the file via path to view it.

---



## Vision modes

Set `agent_params["vision"]` in `main.py`:


| Mode    | Meaning                                              |
| ------- | ---------------------------------------------------- |
| `short` | No bundling — evaluate only the immediate offer      |
| `long`  | Bundling — aggregate value over the planning horizon |




## What a bundled choice entails

A choice is more often between extended experiences than individual events (Ainslie, 2001, pp. 84). The agent is therefore assumed to make a **categorical** choice: as if the current choice is also a commitment to making similar choices in the future.

Whether that commitment is deterministic or stochastic depends on the stickiness parameter `s`:

- Without stickiness (or `s = 1`): long vision is pure commitment
- With `0 ≤ s < 1`: future actions may switch with positive probability (`sticky_decisions.py`)



## Core contract

- Agent talks to the env only via `get_actions` / `simulate`
- Env advances time with `step`
- Discounting lives only in the agent
- Sticky path replaces pure aggregate with weighted SS/LL mixes via `stick_p_t(s, t)`; at `s = 1` it matches `HypAgent` long vision



### Layers


| Layer          | Files                                               | Role                                                            |
| -------------- | --------------------------------------------------- | --------------------------------------------------------------- |
| Entry / params | `main.py` · `sticky_decisions.py`                   | Owns `env_params`, `agent_params`, `max_steps`; sticky adds `s` |
| Core           | `environments/timeline.py` · `agents/hyperbolic.py` | Timeline offers SS/LL; HypAgent discounts and chooses           |
| Outcomes       | main run loop · sticky · `visual_demos/*`           | Independent products that call core; never redefine it          |


---



## Read order

1. `main.py` — params + canonical day loop
2. `environments/timeline.py` — what a day is (hours, SS/LL, memoryless)
3. `agents/hyperbolic.py` — values and choices (short vs long)
4. `sticky_decisions.py` — soft commitment (`s < 1`) on the same core
5. `visual_demos/*` — illustrations; none change the core

Skip `__init__.py` files (package markers only).

## Runtime order

1. Load params from `main.py`
2. Build `TimelineEnvironment` + `HypAgent`
3. `reset()` → state = start time (hours)
4. Each decision day: `get_actions` → evaluate (short: `simulate` / long: `horizon_outcomes`) → greedy `choose_action` → `step(+decision_interval)`
5. Stop when env finished or `max_steps` hit

---



## Quick start

```bash
cd "src/simplified decision agent"
python main.py
```

Sticky valuation at the first decision node (no env stepping):

```bash
python -m sticky_decisions
```

---



## Visual demos

Each demo is independent: own entrypoint, own question. They share params/core but do not depend on each other. Run from this package root.


| Demo                      | Description                                                                                                                                                                    | Run                                                                    |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------- |
| `hyperbolic_curve.py`     | Very simplified visualisation of how a hyperbolically discounted reward looks as the agent approaches its delivery time.                                                       | `python visual_demos/hyperbolic_curve.py`                              |
| `visualise_sails.py`      | Shows the choices the agent imagines/simulates during its current instance of decision making. Subsequent choice values are discounted to the current decision-node time step. | `python -m visual_demos.visualise_sails`                               |
| `aggregate_sail.py`       | Shows the effective sail that the observer currently sees: the aggregate of all discounted values across their vision horizon.                                                 | `python -m visual_demos.aggregate_sail`                                |
| `visualizing_main.py`     | Same as `visualise_sails`, except the user can step through the environment. Accumulated rewards are not tallied or considered (stateless agent).                              | `python -m visual_demos.visualizing_main`                              |
| `sticky_graph.py (WIP)`   | For each stickiness `s` and LL delay, the maximum SS that can still be resisted (indifference frontier).                                                                       | `python -m visual_demos.sticky_graph`                                  |
| `sticky_web_app.py (WIP)` | Same frontier as `sticky_graph`, with live sliders for delay range and LL magnitude.                                                                                           | `python -m visual_demos.sticky_web_app` → open `http://127.0.0.1:5000` |




## Learning paths


| Goal                 | Path                                                                           |
| -------------------- | ------------------------------------------------------------------------------ |
| Decision loop        | `main.py` → `timeline.py` → `hyperbolic.py` → (optional) `visualizing_main.py` |
| Discounting visually | `hyperbolic_curve.py` → `visualise_sails.py` / `aggregate_sail.py`             |
| Stickiness           | `sticky_decisions.py` → `sticky_graph.py` → `sticky_web_app.py`                |




## Who owns what


| Owner            | Owns                                              |
| ---------------- | ------------------------------------------------- |
| Timeline         | Time and offers (SS now / LL later)               |
| HypAgent         | Perceived value and greedy choice (short vs long) |
| main             | Closed-loop day simulation                        |
| sticky_decisions | Soft commitment valuation at one node             |
| visual_demos     | Explanations; call core, never redefine it        |


Sticky valuation is a snapshot at the first decision node — it does not step the environment. Baseline comparison in `sticky_decisions` uses unchanged `HypAgent` (implicit `s = 1`).