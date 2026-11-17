# Visual demos: project scripts vs web app

Side-by-side of `visual_demos/` (+ `animated demo/`) and `web_app/`.  
**Same** = mechanism and intent match. **To be aligned** = one side has it, the other doesn’t (or layout/math presentation diverges).

| Aspect | Project (`visual_demos/`) | Web app (`web_app/`) |
| --- | --- | --- |
| Rendering | Matplotlib windows | Plotly in browser (math in `plot_series.py`) |
| Params | Per-script / `main.py` + `agent_configs` | Shared live panel (SS/LL, length, horizon, k, vision, max steps) |
| Sticky knobs | `sticky_params.perceived_s`, `learn_stickiness` (`agents/sticky_agent.py`) | Sticky sails use fixed `perceived_s` (UI still labeled “Stickiness s”; API query `?stickiness=`) |
| Interaction | Click / keypress / static `plt.show` | Tabs + Step/Reset + sliders / click-to-set `s` |

---

## Coverage map

| Mechanism                                      | Project script                                  | Web section                       | Status                                                  |
| ---------------------------------------------- | ----------------------------------------------- | --------------------------------- | ------------------------------------------------------- |
| Stepped day-by-day sails (HypAgent + env.step) | `visualizing_main.py`                           | Steppable → `visualizing_main`    | Keep web-app version                                    |
| Static choice sails + Σ stack                  | `visualise_sails.py`                            | Non-steppable → `visualise_sails` | Same (minor visual quirks)                              |
| Bundled resist sail + snack aggregate          | `aggregate_sail.py`                             | Non-steppable → `aggregate_sail`  | Same (minor visual quirks)                              |
| Reward approaching t=0 (fixed agent)           | `hyperbolic_curve.py`                           | Steppable → `hyperbolic_curve`    | Same mechanism; **to be aligned** (params source)       |
| Max SS resisted vs stickiness                  | `sticky_graph.py`                               | Sticky → top chart                | Same maths; minor design differences for visual clarity |
| Sticky sails (sticky EV aggregates)            | `visualise_sticky_sails.py`                     | Sticky → bottom chart             | Same maths; Still figuring out the better visual        |
| Sticky foresight “beaker” panels               | `sticky_foresight_panels.py` (via sticky sails) | —                                 | **To be aligned** (project only)                        |
| Stepped sticky + Bayesian belief               | `sticky_step.py`                                | —                                 | **To be aligned** (project only)                        |
| Animated sticky path mass                      | `animated demo/animate_trajectories.py`         | —                                 | **To be aligned** (project only)                        |

---

## Per-demo notes

### 1. `visualizing_main` — stepped hyperbolic sails

| | Project | Web |
| --- | --- | --- |
| Mechanism | Same: each step `choose_action` → `environment.step`; redraw foresight sails | Same |
| Layout | **Single** sails axes (no Σ stack) | **Dual** panel via `sails_frame` (Σ stack + sails) — same layout as static `visualise_sails` |
| Advance | Click / keypress | Step button |
| Cap | `main.max_steps` | Same (`max_steps` in shared config) |
| Console | Prints per-day values + LL−SS gaps | Meta line only (choice + snack/resist) |

**Keep web-app version:** dual-panel stepped sails is the reference; project script is still single-panel.

---

### 2. `visualise_sails` — static sails + aggregate stack

| | Project | Web |
| --- | --- | --- |
| Mechanism | Same: HypAgent foresight sails; left Σ = sum of per-day V_hyp | Same (`sails_frame`) |
| Visual quirks | Matplotlib Reds/Blues colormaps for stack segments | Horizon `fade()` toward white on snack/resist hexes |
| Labels / choice | Title + bold aggregate numeral | Title HTML bold + meta line |

**Same** overall. Stack colour ramp is a minor visual difference only.

---

### 3. `aggregate_sail` — bundled sail

| | Project | Web |
| --- | --- | --- |
| Mechanism | Same: resist height = sum of still-outstanding LL discounts; snack = one t=0 aggregate | Same (`aggregate_frame`) |
| Choice emphasis | Standard line/markers | Thicker chosen sail / larger markers / ★ legend / bold numerals |

**Same** mechanism. Chosen-action emphasis is stronger on web (cosmetic).

---

### 4. `hyperbolic_curve` — approach to delivery

| | Project | Web |
| --- | --- | --- |
| Mechanism | Same: agent fixed at 0; sail shortens as delay → 0; fixed axis ranges | Same (`hyperbolic_frame`) |
| Params | Hardcoded `LL={value:10, delay:10}`, `discount=0.8` | Shared panel LL value/delay + discount factor `k` |
| Advance | Auto loop + waitforbuttonpress | Step / Reset |

**Same** mechanism. **To be aligned:** project script ignores live env/agent config; web is driven by shared params.

---

### 5. Sticky pair — `sticky_graph` + `visualise_sticky_sails`

| | Project | Web |
| --- | --- | --- |
| Sticky solve | Same `max_ss_resisted` / `stick_p_t` | Same (duplicated in `plot_series` so sticky_graph side effects aren’t imported) |
| Graph UX | All delay curves same style; fixed `ylim(0, 8)`; edit `ll_delays` in source | Current LL delay in **black**; click black curve → set `s`; delay max/step sliders; y zoom around current curve |
| Sticky sails layout | **Single** sails axes; opaque t=0 sticky aggregates under fixed `perceived_s` (`learn_stickiness=False`) | **Dual** panel: sticky EV Σ stack + sails |
| Extra figure | Optional foresight beaker panels (`SHOW_STICKY_BREAKDOWN`) | Explicitly omitted (`plot_series.sticky_sails_frame`) |
| Stickiness control | Edit `sticky_params["perceived_s"]` | Number input + graph click (maps to agent `perceived_s`) |

**Same** core sticky maths. **Can be aligned:**

- Graph interaction / current-curve highlight / y-scale (web-only today).
- Sticky sails dual Σ stack (web) vs single panel (project).
- Foresight breakdown panels (project-only).

---

### 6. Project-only (no web counterpart)

| Script | What it shows (current behaviour) |
| --- | --- |
| `sticky_foresight_panels.py` | Visit-probability beaker tracks for snack-first vs resist-first |
| `sticky_step.py` | Stepped sticky sails (left) + Beta posterior (right). `StickyTimelineEnvironment` owns `beta_belief` and always updates ŝ; `learn_stickiness` only gates whether the agent’s foresight uses env ŝ vs fixed `perceived_s`. Intent vs realized action from stick/switch rolls. |
| `animated demo/animate_trajectories.py` | Animated accumulation of snack-first path mass → `stick_p_t` |

**To be aligned:** none of these exist in the web app.

---

### 7. Web-only conveniences (no single project script)

| Feature | Notes |
| --- | --- |
| Shared live config + validation | Infinite length ↔ finite horizon rules, max steps, vision toggle |
| Sticky section as stacked pair | Graph + sails linked by `perceived_s` |
| Reset to default | Seeds from `main` / `agent_configs` |

**To be aligned** only if the matplotlib scripts should share the same interactive param surface (optional; not required for mechanism parity).

---

## Naming / module notes (current)

| Item | Current |
| --- | --- |
| Sticky agent module | `agents/sticky_agent.py` (`StickyHypAgent`) |
| Perceived stickiness param | `perceived_s` in `sticky_params` / agent ctor |
| Learning toggle | `learn_stickiness` — env always updates belief when attached; agent only tracks ŝ when True |
| True stickiness | `env_params["true_stickiness"]` on `StickyTimelineEnvironment` |

---

## Quick verdict

- **Mechanisms that match:** sails maths, aggregate bundle, hyperbolic approach, sticky threshold solve, sticky EV aggregates (fixed `perceived_s`).
- **Largest gaps to align:**    
  1. Sticky foresight'beaker' panels (project only)  
  2. Sticky learning / animation demos (`sticky_step`, `animate_trajectories`) missing from web  
- **Decided / open (not hard gaps):** keep web dual-panel for stepped sails; sticky graph = minor design differences; sticky sails visual still being figured out.
