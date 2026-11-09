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



## Architecture Core

- Agent talks to the env only via `get_actions` / `simulate`
- Env advances time with `step`
- Discounting lives only in the agent under `value_function`
- Sticky path replaces pure aggregate with weighted SS/LL mixes via `stick_p_t(s, t)`; at `s = 1` it matches `HypAgent` long vision



### Layers


| Layer         | Files                                                        | Role                                                                   |
| ------------- | ------------------------------------------------------------ | ---------------------------------------------------------------------- |
| Config Params | `main.py` · `sticky_decisions.py`                            | Owns `env_params`, `agent_params`, `max_steps`; sticky adds `s`        |
| Logic Core    | `environments/timeline.py` · `agents/hyperbolic.py`          | Timeline offers SS/LL; HypAgent discounts and chooses                  |
| Outcomes      | main run loop · sticky_decisions run loop · `visual_demos/*` | Independent products that call core logic to observe agent's decisions |
| Web UI        | `web_app/demo_web_app.py` · `web_app/plot_series.py`         | Flask + Plotly; series builders mirror visual demos                    |


---

## Read order

1. `environments/timeline.py` — environment defaults and available actions/paths within environment
2. `agents/hyperbolic.py` — value function and decision-making logic
3. `main.py` — config parameters; run to view decision computations (printout values)
4. `sticky_decisions.py` — adds stickiness parameter (uses same agent and environment)
5. `visual_demos/*` — illustrations with varying parameters and contexts for demonstration
6. `web_app/*` — unified browser UI over the demos (`plot_series` builds Plotly payloads)

`__init__.py` files  are just package markers only

## Runtime order ([main.py](http://main.py))

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

Each demo is an independent visualisation to demonstrate different layers/variations of the effect. They share params/core but do not depend on each other. Run from this package root.


| Demo                      | Description                                                                                                                                                                    | Run                                                 |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------- |
| `hyperbolic_curve.py`     | Very simplified visualisation of how a hyperbolically discounted reward looks as the agent approaches its delivery time.                                                       | `python visual_demos/hyperbolic_curve.py`           |
| `visualise_sails.py`      | Shows the choices the agent imagines/simulates during its current instance of decision making. Subsequent choice values are discounted to the current decision-node time step. | `python -m visual_demos.visualise_sails`            |
| `aggregate_sail.py`       | Shows the effective sail that the observer currently sees: the aggregate of all discounted values across their vision horizon.                                                 | `python -m visual_demos.aggregate_sail`             |
| `visualizing_main.py`     | Same as `visualise_sails`, except the user can step through the environment. Accumulated rewards are not tallied or considered (stateless agent).                              | `python -m visual_demos.visualizing_main`           |
| `sticky_graph.py`         | For each stickiness `s` and LL delay, the maximum SS that can still be resisted (indifference frontier).                                                                       | `python -m visual_demos.sticky_graph`               |
| `web_app/demo_web_app.py` | Unified UI: steppable / non-steppable / sticky demos with shared param controls.                                                                                               | See [Unified web app](#unified-web-app) below |




### Unified web app

Interactive browser UI over the demos above. Same venv and deps as [Quick start](#quick-start) (`flask` / `gunicorn` are in `requirements.txt`).

#### Local

From the **repo root**:

```bash
python3.11 -m venv .venv          # once, if .venv is missing
source .venv/bin/activate
pip install -r requirements.txt   # once, or after requirements change
```

Then from the package directory:

```bash
cd "src/simplified decision agent"
python -m web_app.demo_web_app
```

Open **[http://127.0.0.1:5050](http://127.0.0.1:5050)** in Firefox, Chrome, or Safari (use `127.0.0.1`, not `localhost` — on macOS, `localhost:5000` is often AirPlay and returns 403). Stop the server with `Ctrl+C`.

#### Hosted (Render)

This repo includes a `render.yaml` Blueprint. On [Render](https://render.com): **New → Blueprint** (or **Web Service**), connect this GitHub repo, and deploy. Free tier sleeps when idle; the first visit after sleep can take ~30–60s.

Manual Web Service settings if you skip the Blueprint:

- **Build command:** `pip install -r requirements.txt`
- **Start command:** `gunicorn -b 0.0.0.0:$PORT --chdir "src/simplified decision agent" web_app.demo_web_app:app`

After deploy, put the public URL here (and optionally at the top of this README):

```text
Live demo: https://YOUR-SERVICE.onrender.com
```

**Shared params** (apply to every tab): SS/LL rewards and delays, env length, agent horizon, discount factor `k`, max steps, short/long vision. Clear length or horizon for *none* (at least one must be set). Blank max steps = no run cap (same as `main.max_steps = None`).

**Tabs:**


| Tab           | Demos                                   | Controls                                             |
| ------------- | --------------------------------------- | ---------------------------------------------------- |
| Steppable     | `visualizing_main` · `hyperbolic_curve` | Reset / Step — env days, or reward approaching day 0 |
| Non-steppable | `visualise_sails` · `aggregate_sail`    | Demo select only                                     |
| Sticky        | sticky indifference frontier            | Extra LL delay max / step sliders                    |


Plotly loads from a CDN (needs network on first load). Math lives in `web_app/plot_series.py` (mirrors the matplotlib demos); the browser only draws.