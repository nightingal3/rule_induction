import backoff
import logging
import openai
from typing import Union, List, Optional, Tuple, Callable
import src.globals
import socket
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

from text_generation import Client
from typing import Callable

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
    model_name: str = "gpt-3.5-turbo",
    temp: float = 0.7,
    logprobs: bool = False,
    max_tokens: Optional[int] = None,
) -> Tuple[str, float, float]:

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
        src.globals.TOTAL_COST += (
            prompt_tokens / 1000
        ) * src.globals.OPENAI_CURRENT_COSTS[model_name]["prompt"]
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
        src.globals.TOTAL_COST += (
            num_prompt_tokens / 1000
        ) * src.globals.OPENAI_CURRENT_COSTS[model_name]["prompt"] + (
            num_completion_tokens / 1000
        ) * src.globals.OPENAI_CURRENT_COSTS[
            model_name
        ][
            "completion"
        ]

    return completion


def make_completion_fn_tgi_server(model_name: str) -> Callable:
    models = Client.list_from_central()
    model_addr = [m["address"] for m in models if m["name"] == model_name]
    assert len(model_addr) > 0, f"Model {model_name} not found"
    client = Client("http://" + model_addr[0])

    @backoff.on_exception(
        backoff.constant,
        (socket.timeout, socket.error),
        max_tries=30,
        on_backoff=backoff_printer,
        interval=5,
    )
    def get_completion_open_llm(
        prompt: str,
        temp: str = 1.0,
        return_details: bool = False,
        max_new_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        nonlocal client
        if isinstance(prompt, list):
            prompt_text = prompt[1]["content"]  # user prompt
        else:
            prompt_text = prompt
        output = client.generate(prompt_text, temperature=temp)
        if return_details:
            return output

        return output.generated_text

    return get_completion_open_llm


def make_completion_fn_hf(model_name: str) -> Callable:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    generator = pipeline("text-generation", model=model_name, device=0)

    @backoff.on_exception(
        backoff.constant,
        all_error_types,
        max_tries=30,
        on_backoff=backoff_printer,
        interval=5,
    )
    def get_completion_hf(prompt: str, temp: str = 1.0) -> str:
        generated_text = generator(prompt, temp=temp, num_return_sequences=1)
        # input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        # output = model.generate(input_ids, temperature=temp)
        # return tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text

    return get_completion_hf
