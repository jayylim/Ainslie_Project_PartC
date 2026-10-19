### **Environment**

> The environment is just there to lay out the series of subsequent decisions for the agent. Each 'choice node' is ultimately still a choice made in the 'present' (at time step of said choice)
> For this simulation, the agent is on a diet. Once a day, the agent chooses between 'snacking' or resisting. The  LL received is the utility of weighing in the next morning and not gaining weight. 

**Properties**
- Timeline of sequential decisions with recurring decision nodes between a shorter, sooner (immediate) reward and large, later reward.
- Each decision is exactly the same (fixed magnitude and delay for both SS and LL)
- Timescale: 1 timestep = 1 hour; decisions every `decision_interval` hours (default 24)
- LL delay is hours relative to the choice (e.g. 12)
- To be more representative of reality, the agent aggregates rewards over an extended period of time as if making a **bundled choice** *(Ainslie, 2001, pp. 84)*
- Deterministic environment
- Environment does not care about choice history (currently)
- doesn't care about the reward he HAS received, and only looks at each individual decision.

### Stickiness
Let the stickiness parameter (*s*) be P(decision_{t+1} = decision_t) for all t



### 13 Oct Brain Storm

- The agent can either have short-sighted (where it never aggregates, or effectively has vision = 1) or be long-sighted (where it aggregates over a set vision length)
  - If the environment is infinite, then the vision length takes priority
  - If the environment has a fixed length, then the vision length caps at max duration. (If set to none, it will always take the max available length).
    - Right now, this means that as long as the environment has a horizon greater than the max allowed processing (including the agent's vision), values computed for every decision will be identitcal. Is there a way to change this? This should make sense from real-life perspective as well.


environment horizon can be infinite
- human vision horizon is defnitely finite
- the person sees rewards that are a constant distance away => perceived value of options are always the same?

environment has finite horizon and agent's vision is equal or greater to 
- values will be shrinking towards the end of the environment timeline
pay off (deadline/performance for practicing instrument)

If you choose for 20 nights, you see the aggregate
If you choose for just 1 night, you see the identical decision


now based on the current environment, agent and run code, i want to add a visualisation of the experiment that is run by @src/simplified decision agent/main.py. The aggregated SS and LL values should be present as a hyperbolic curve sail increasing towards positive x (where x axis is time to reward)

#### Redefining the environment
**environmen definition**
- an environment where you are in pursuit of an LL but faced with SS every decision node
- the LL is of a fixed value and delay, but is repeatedly offered to you in the future. This means that you face the prospect of multiple delayed rewards. 
  - real life example is the weighing yourself in the morning vs the cookie now
- other environments have a single delayed payoff: e.g. a music recital. Even so, choosing to practice for said recital might involve multiple delayed subrewards that constitute sequential LLs? Or is this a different environment altogether

**results right now**
 - I am only modelling a categorical decision now: what would you choose assuming this decision would last for the forseeable future
  - When you repeat a categorical decision every day in a **finite environment**, you are actually makign a different decision daily because the time horizon is shrinking 
    - so its subject to your imagination of the (cumulative) LL
  - if its **infinite**, then you are just repeating the categorical deciiosn over and over again, and values will always be the same (unless your parameters change)
    - the exact values are also subject to how much you can see
  
- for modelling an instantaneous decision with a singular fixed LL, if the environment is infinite, then perceived value of LL doesnt change bcasue delay is same (which is subject to what you can see) -> if you are stateless, all values are the same
  - thats why agent cannot be stateless if they want to overcome temptation


**NEED TO CHANGE**
- have CATEGORICAL MODE and INSTANTANEOUS MODE -> this is to show that categorical decisions will lead to **different perceived values** of each option (and perhaps a different choice made)
- STATELESS VS MEMORY ENVIRONMENT
  - this is where you bring in stickiness parameter?
