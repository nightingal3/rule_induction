from text_generation import Client
import argparse
from typing import Callable
import logging

from prompt_openai import do_task, init_task, make_finish_task
from common_args import get_common_arguments


def make_completion_fn(model_name: str) -> Callable:
    models = Client.list_from_central()
    model_addr = [m["address"] for m in models if m["name"] == model_name]
    assert len(model_addr) > 0, f"Model {model_name} not found"
    client = Client("http://" + model_addr[0])

    def get_completion_open_llm(
        prompt: str,
        temp: str = 1.0,
        **kwargs,
    ) -> str:
        nonlocal client
        if isinstance(prompt, list):
            prompt_text = prompt[1]["content"]  # user prompt
        else:
            prompt_text = prompt
        output = client.generate(prompt_text, temperature=temp)
        return output.generated_text

    return get_completion_open_llm


if __name__ == "__main__":
    parser = get_common_arguments()
    parser.add_argument(
        "--model",
        type=str,
        default="lmsys/vicuna-7b-v1.5",
        choices=["lmsys/vicuna-7b-v1.5", "meta-llama/Llama-2-70b-chat-hf"],
        help="model to use (please look at list of models in cache)",
    )
    # parser.add_argument(
    #     "--temp", default=0.0, type=float, help="Temperature for sampling"
    # )
    # parser.add_argument(
    #     "--dataset", required=True, choices=["scan", "cogs", "colours", "functions"]
    # )
    # parser.add_argument(
    #     "--split",
    #     default="simple",
    #     choices=[
    #         "simple",
    #         "length",
    #         "jump",
    #         "cp_recursion",
    #         "prim_to_subj_common",
    #         "exposure_example_obj_proper",
    #         "obj_to_subj_common",
    #         "only_seen_as_unacc_subj_as_obj_omitted_transitive_subj",
    #     ],
    # )
    # parser.add_argument(
    #     "--prompt_type",
    #     default="base",
    #     choices=["base", "full_grammar", "grammar_induction", "rule_selection"],
    # )
    # parser.add_argument("--output", type=str)
    # parser.add_argument("--start_ind", type=int)
    # parser.add_argument("--end_ind", type=int)
    # parser.add_argument("--num_few_shot_examples", type=int, default=5)
    # parser.add_argument("--use_min_cover", action="store_true")
    # parser.add_argument("--prompt_in_loop", action="store_true")
    # parser.add_argument("--return_induced_grammar_only", action="store_true")
    # parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    output_file = (
        args.output
        if args.output is not None
        else f"./logs/{args.dataset}_{args.split}_{args.prompt_type}_{args.model}_{args.start_ind}_{args.end_ind}_minset_{args.use_min_cover}.csv"
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO if not args.debug else logging.DEBUG)
    args = parser.parse_args()
    start_ind = args.start_ind if args.start_ind is not None else 0

    lm_completion_fn = make_completion_fn(args.model)
    finish_task = make_finish_task(
        args, output_file, args.return_induced_grammar_only, start_ind
    )

    try:
        task = init_task(args)
        start_ind = args.start_ind if args.start_ind is not None else 0
        end_ind = args.end_ind if args.end_ind is not None else len(task)
        acc, results_log = do_task(
            task,
            args.model,
            args.prompt_type,
            args.temp,
            start_ind=start_ind,
            end_ind=end_ind,
            get_grammar_only=args.return_induced_grammar_only,
            get_completion_fn=lm_completion_fn,
        )
        finish_task(
            args,
            acc,
            results_log,
            output_file,
            get_grammar_only=args.return_induced_grammar_only,
        )
    except ValueError:
        raise ValueError(f"Dataset {args.dataset} not registered")
