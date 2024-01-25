# This is kind of ugly. But keeping costs in a global variable for now.
TOTAL_COST = 0

# per 1k tokens
OPENAI_CURRENT_COSTS = {
    "gpt-4-1106-preview": {
        "completion": 0.03,
        "prompt": 0.01,
    },
    "gpt-4": {
        "completion": 0.06,
        "prompt": 0.03,
    },
    "gpt-3.5-turbo-0613": {"completion": 0.002, "prompt": 0.001},
    "davinci-002": {"completion": 0.012, "prompt": 0.012},
}
