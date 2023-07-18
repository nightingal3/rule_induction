import llm_client # uses lti-llm deployment lib
import argparse
from typing import Callable

from utils.utils import *
from prompt_openai import do_task, init_task, finish_task

def make_completion_fn(client: llm_client.Client) -> Callable:
    def get_completion_open_llm(prompt: str, temp: str, return_scores: bool = False, return_hidden_states: bool = False) -> str:
        nonlocal client
        prompt_text = prompt[1]["content"] # user prompt
        output = client.prompt(prompt_text, temp=temp, return_scores=return_scores, return_hidden_states=return_hidden_states, max_new_tokens=512)
        return output

    return get_completion_open_llm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prompt open source (huggingface models) with task specs")
    parser.add_argument("--model", type=str, default="llama-65b", choices=["llama-65b", "llama-30b", "alpaca"], help="model to use (please look at list of models in cache)")
    parser.add_argument("--node", type=str, help="Which tir node the model is running on")
    parser.add_argument("--temp", default=0.0, type=float, help="Temperature for sampling")
    parser.add_argument("--dataset", required=True, choices=["scan", "cogs"])
    parser.add_argument("--split", default="simple", choices=["simple", "length", "jump", "cp_recursion", "prim_to_subj_common", "exposure_example_obj_proper", "obj_to_subj_common", "only_seen_as_unacc_subj_as_obj_omitted_transitive_subj"])
    parser.add_argument("--prompt_type", default="base", choices=["base", "full_grammar", "grammar_induction"])
    parser.add_argument("--output", type=str)
    parser.add_argument("--start_ind", type=int)
    parser.add_argument("--end_ind", type=int)
    parser.add_argument("--num_few_shot_examples", type=int, default=5)
    parser.add_argument("--use_min_cover", action="store_true")
    parser.add_argument("--return_induced_grammar_only", action="store_true")

    args = parser.parse_args()
    output_file = args.output if args.output is not None else f"./logs/{args.dataset}_{args.split}_{args.prompt_type}_{args.model}_{args.start_ind}_{args.end_ind}_minset_{args.use_min_cover}.csv"

    job_id, lm_node = args.node if args.node is not None else get_job_info_by_name(f"lti-llm-{args.model}")

    lm_client = llm_client.Client(address=lm_node)
    lm_completion_fn = make_completion_fn(lm_client)
    try:
        task = init_task(args)
        start_ind = args.start_ind if args.start_ind is not None else 0
        end_ind = args.end_ind if args.end_ind is not None else len(task)
        acc, results_log, total_completion_tokens, total_prompt_tokens = do_task(task, args.model, args.prompt_type, args.temp, start_ind=start_ind, end_ind=end_ind, get_grammar_only=args.return_induced_grammar_only, get_completion_fn=lm_completion_fn)
        finish_task(args, acc, results_log, output_file, get_grammar_only=args.return_induced_grammar_only)
    except ValueError:
        raise ValueError(f"Dataset {args.dataset} not registered")


        

