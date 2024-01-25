import backoff
import logging
import openai
from typing import Union, List, Optional, Tuple
from src.globals import OPENAI_CURRENT_COSTS, TOTAL_COST

all_error_types = (
    openai.error.OpenAIError,
    openai.error.APIError,
    openai.error.RateLimitError,
    openai.error.APIConnectionError,
    openai.error.ServiceUnavailableError,
)


def backoff_printer(details):
    logging.info(
        f"Backing off {details['wait']} seconds after {details['tries']} tries calling function {details['target'].__name__} with args {details['args']} and kwargs {details['kwargs']}"
    )


@backoff.on_exception(
    backoff.constant,
    all_error_types,
    max_tries=30,
    on_backoff=backoff_printer,
    interval=5,
)
def get_completion_openai(
    prompts: Union[List, str],
    model_name: str,
    temp: float = 0.7,
    logprobs: bool = False,
    max_tokens: Optional[int] = None,
) -> Tuple[str, float, float]:
    global TOTAL_COST

    if max_tokens == 0:
        # Just scoring the prompt - use the old completions endpoint
        completion = openai.Completion.create(
            model=model_name,
            prompt=prompts,
            temperature=temp,
            logprobs=1,
            max_tokens=max_tokens,
            echo=True,
        )
        prompt_tokens = completion["usage"]["total_tokens"]
        TOTAL_COST += (prompt_tokens / 1000) * OPENAI_CURRENT_COSTS[model_name][
            "prompt"
        ]
    else:
        completion = openai.ChatCompletion.create(
            model=model_name,
            messages=prompts,
            temperature=temp,
            logprobs=logprobs,
            max_tokens=max_tokens,
        )
        num_prompt_tokens = completion["usage"]["prompt_tokens"]
        num_completion_tokens = completion["usage"]["completion_tokens"]
        TOTAL_COST += (num_prompt_tokens / 1000) * OPENAI_CURRENT_COSTS[model_name][
            "prompt"
        ] + (num_completion_tokens / 1000) * OPENAI_CURRENT_COSTS[model_name][
            "completion"
        ]

    return completion
