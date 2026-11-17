# Visual demos: project scripts vs web app

Side-by-side of `visual_demos/` (+ `animated demo/`) and `web_app/`.  
**Same** = mechanism and intent match. **To be aligned** = one side has it, the other doesn’t (or layout/math presentation diverges).

| Aspect | Project (`visual_demos/`) | Web app (`web_app/`) |
| --- | --- | --- |
| Rendering | Matplotlib windows | Plotly in browser (math in `plot_series.py`) |
| Params | Per-script / `main.py` + `agent_configs` | Shared live panel (SS/LL, length, horizon, k, vision, max steps) |
| Interaction | Click / keypress / static `plt.show` | Tabs + Step/Reset + sliders / click-to-set `s` |

---

## Coverage map

| Mechanism                                      | Project script                                  | Web section                       | Status                                                  |
| ---------------------------------------------- | ----------------------------------------------- | --------------------------------- | ------------------------------------------------------- |
| Stepped day-by-day sails (HypAgent + env.step) | `visualizing_main.py`                           | Steppable → `visualizing_main`    | **To be aligned** (layout)                              |
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

**To be aligned:** web steps a dual-panel view; project script is single-panel. Either drop the stack from steppable web, or add the stack to `visualizing_main.py`.

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
| Sticky sails layout | **Single** sails axes; opaque t=0 sticky aggregates | **Dual** panel: sticky EV Σ stack + sails (aligned with `visualise_sails` look) |
| Extra figure | Optional foresight beaker panels (`SHOW_STICKY_BREAKDOWN`) | Explicitly omitted (`plot_series.sticky_sails_frame` docstring) |
| Stickiness control | Edit `sticky_params` / script constant | Number input + graph click |

**Same** core sticky maths. **To be aligned:**

- Graph interaction / current-curve highlight / y-scale (web-only today).
- Sticky sails dual Σ stack (web) vs single panel (project).
- Foresight breakdown panels (project-only).

---

### 6. Project-only (no web counterpart)

| Script | What it shows |
| --- | --- |
| `sticky_foresight_panels.py` | Visit-probability beaker tracks for snack-first vs resist-first |
| `sticky_step.py` | Stepped sticky sails + Beta belief over `s`; `StickyTimelineEnvironment` + Bayesian updates |
| `animated demo/animate_trajectories.py` | Animated accumulation of snack-first path mass → `stick_p_t` |

**To be aligned:** none of these exist in the web app.

---

### 7. Web-only conveniences (no single project script)

| Feature | Notes |
| --- | --- |
| Shared live config + validation | Infinite length ↔ finite horizon rules, max steps, vision toggle |
| Sticky section as stacked pair | Graph + sails linked by `s` |
| Reset to default | Seeds from `main` / `agent_configs` |

**To be aligned** only if the matplotlib scripts should share the same interactive param surface (optional; not required for mechanism parity).

---

## Quick verdict

- **Mechanisms that match:** sails maths, aggregate bundle, hyperbolic approach, sticky threshold solve, sticky EV aggregates.
- **Largest gaps to align:**  
  1. Stepped sails layout (single vs dual panel)
  2. Sticky sails layout + foresight panels
  3. Sticky graph UX (highlight / click-`s` / y-scale)
  4. Entire sticky learning / animation demos (`sticky_step`, foresight panels, `animate_trajectories`) missing from web
  5. `hyperbolic_curve.py` hardcoded params vs web shared config
