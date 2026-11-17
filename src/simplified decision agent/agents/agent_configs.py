agent_params = {
    "discount_factor": 0.6,
    "vision": "long",  # "short" or "long"
    "horizon": 10,  # max # of decisions to plan; capped by remaining env decisions, includes the current day (min. 1)
}

sticky_params = {
    "discount_factor": 0.6,
    "vision": "long",  # "short" or "long"
    "horizon": 10,  # max # of decisions to plan; capped by remaining env decisions, includes the current day (min. 1)
    "perceived_s": 0.6,
    "learn_stickiness": True, # will be a toggle in most demos for learned stickiness to matter
}