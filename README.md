# Simplified Decision Agent

Hyperbolic discounter on a memoryless SS/LL timeline (Ainslie-style bundling). Edit params in `main.py`, then run from this directory.

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



## Archtecture Core

- Agent talks to the env only via `get_actions` / `simulate`
- Env advances time with `step`
- Discounting lives only in the agent under `value_function`
- Sticky path replaces pure aggregate with weighted SS/LL mixes via `stick_p_t(s, t)`; at `s = 1` it matches `HypAgent` long vision



### Layers


| Layer          | Files                                               | Role                                                            |
| -------------- | --------------------------------------------------- | --------------------------------------------------------------- |
| Entry / params | `main.py` · `sticky_decisions.py`                   | Owns `env_params`, `agent_params`, `max_steps`; sticky adds `s` |
| Core           | `environments/timeline.py` · `agents/hyperbolic.py` | Timeline offers SS/LL; HypAgent discounts and chooses           |
| Outcomes       | main run loop · sticky · `visual_demos/*`           | Independent products that call core; never redefine it          |


---



## Read order

1. `environments/timeline.py` — type of decision timeline/horizon (hours, SS/LL, memoryless)
2. `agents/hyperbolic.py` — value function and decision-making logic
3. `main.py` — config parameters; run to decision making loop (printout values)
4. `sticky_decisions.py` — adds stickiness parameter (uses same agent and environment)
5. `visual_demos/*` — illustrations with varying parameters

`__init__.py` files  are just package markers only

## Runtime order

1. Load params from `main.py`
2. Build `TimelineEnvironment` + `HypAgent`
3. `reset()` → state = start time (hours)
4. Each decision day: `get_actions` → evaluate (short: `simulate` / long: `horizon_outcomes`) → greedy `choose_action` → `step(+decision_interval)`
5. Stop when env finished or `max_steps` hit

---



## Quick start

Use the project virtualenv (Python 3.11). From the repo root:

```bash
python3.11 -m venv .venv          # once, if .venv is missing
source .venv/bin/activate
pip install -r requirements.txt   # once, or after requirements change
```

Then run from the package directory:

```bash
cd "src/simplified decision agent"
python main.py
```

(With stickiness parameter) Valuation at the first decision node ONLY (no stepping through env):

```bash
python -m sticky_decisions
```

Cursor/VS Code should pick `.venv` automatically via `.vscode/settings.json`. If the wrong interpreter is selected, choose **Python: Select Interpreter** → `.venv/bin/python`.

---


## Visual demos

Each demo is an independent visualiation to demonstrate different layers/variations of the effect . They share params/core but do not depend on each other. Run from this package root.


| Demo                      | Description                                                                                                                                                                    | Run                                                                    |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------- |
| `hyperbolic_curve.py`     | Very simplified visualisation of how a hyperbolically discounted reward looks as the agent approaches its delivery time.                                                       | `python visual_demos/hyperbolic_curve.py`                              |
| `visualise_sails.py`      | Shows the choices the agent imagines/simulates during its current instance of decision making. Subsequent choice values are discounted to the current decision-node time step. | `python -m visual_demos.visualise_sails`                               |
| `aggregate_sail.py`       | Shows the effective sail that the observer currently sees: the aggregate of all discounted values across their vision horizon.                                                 | `python -m visual_demos.aggregate_sail`                                |
| `visualizing_main.py`     | Same as `visualise_sails`, except the user can step through the environment. Accumulated rewards are not tallied or considered (stateless agent).                              | `python -m visual_demos.visualizing_main`                              |
| `sticky_graph.py (WIP)`   | For each stickiness `s` and LL delay, the maximum SS that can still be resisted (indifference frontier).                                                                       | `python -m visual_demos.sticky_graph`                                  |
| `web app/demo_web_app.py` | Unified UI: steppable / non-steppable / sticky demos with shared param controls (rewards, delays, length, horizon, vision).                                                     | See [Unified web app](#unified-web-app-local) below                    |


### Unified web app (local)

Interactive browser UI over the demos above (Steppable / Non-steppable / Sticky tabs). Same venv and deps as [Quick start](#quick-start).

From the **repo root**:

```bash
python3.11 -m venv .venv          # once, if .venv is missing
source .venv/bin/activate
pip install -r requirements.txt   # once, or after requirements change
```

Then from the package directory:

```bash
cd "src/simplified decision agent"
python "web app/demo_web_app.py"
```

Open **http://127.0.0.1:5050** in Firefox, Chrome, or Safari (use `127.0.0.1`, not `localhost` — on macOS, `localhost:5000` is often AirPlay and returns 403). Stop the server with `Ctrl+C`.

Params at the top (SS/LL rewards and delays, env length, horizon, short/long vision) apply to every tab. Sticky has extra LL delay max/step sliders; Plotly loads from a CDN (needs network on first load).


## Learning paths


| Goal                 | Path                                                                           |
| -------------------- | ------------------------------------------------------------------------------ |
| Decision loop        | `main.py` → `timeline.py` → `hyperbolic.py` → (optional) `visualizing_main.py` |
| Discounting visually | `hyperbolic_curve.py` → `visualise_sails.py` / `aggregate_sail.py`             |
| Stickiness           | `sticky_decisions.py` → `sticky_graph.py` → `web app/demo_web_app.py` (Sticky tab) |
| All demos in one UI  | `web app/demo_web_app.py`                                                      |




## Who owns what


| Owner            | Owns                                              |
| ---------------- | ------------------------------------------------- |
| Timeline         | Time and offers (SS now / LL later)               |
| HypAgent         | Perceived value and greedy choice (short vs long) |
| main             | Closed-loop day simulation                        |
| sticky_decisions | Soft commitment valuation at one node             |
| visual_demos     | Explanations; call core, never redefine it        |


Sticky valuation is a snapshot at the first decision node — it does not step the environment. Baseline comparison in `sticky_decisions` uses unchanged `HypAgent` (implicit `s = 1`).