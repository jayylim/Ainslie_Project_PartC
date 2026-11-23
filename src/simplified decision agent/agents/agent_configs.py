agent_params = {
    "discount_factor": 0.6,
    "vision": "long",  # "short" or "long"
    "horizon": 10,  # max # of decisions to plan; capped by remaining env decisions, includes the current day (min. 1)
}

sticky_params = {
    "discount_factor": 0.6,
    "vision": "long",  # "short" or "long"
    "horizon": 10,  # max # of decisions to plan; capped by remaining env decisions, includes the current day (min. 1)
    # Fixed preset when learn_stickiness is Off (works as prior)
    "perceived_s": 0.6,
    # Off → preset foresight s; On → foresight s tracks Beta posterior mean
    # from the agent's stick/switch choices (intent always realized).
    "learn_stickiness": True,
}