import argparse
import openai
import os
from tqdm import tqdm
import backoff
from typing import List, Callable
import pandas as pd
import jsonlines
import datetime
import pprint

from src import get_task

def backoff_printer(details):
    print(f"Backing off {details['wait']} seconds after {details['tries']} tries calling function {details['target'].__name__} with args {details['args']} and kwargs {details['kwargs']}")
    
@backoff.on_exception(backoff.constant, openai.error.OpenAIError, max_tries=30, on_backoff=backoff_printer, interval=5)
def get_completion_openai(prompts: List, model_name: str, temp: float = 0.7) -> str:
    completion = openai.ChatCompletion.create(
        model=model_name,
        messages=prompts,
        temperature=temp,
    )
    return completion
    
def init_task(args: argparse.Namespace):
    _task = get_task(args.dataset)
    if args.dataset == "scan":
        if args.split == "simple":
            train_file = "./data/scan/scan_simple_train.csv"
            test_file = "./data/scan/scan_simple_test.csv"
        elif args.split == "length":
            train_file = "./data/scan/scan_length_train.csv"
            test_file = "./data/scan/scan_length_test.csv"
        elif args.split == "jump":
            train_file = "./data/scan/scan_jump_train.csv"
            test_file = "./data/scan/scan_jump_test.csv"
        else:
            raise ValueError(f"Split {args.split} not registered")
        task = _task(train_file, test_file, prompt_style=args.prompt_type, split=args.split, few_shot_min_set=args.use_min_cover, num_few_shot_examples=args.num_few_shot_examples)
    elif args.dataset == "cogs":
        train_file = "./data/cogs/train_100.tsv"
        test_file = "./data/cogs/gen.tsv"
        task = _task(train_file, test_file, prompt_style=args.prompt_type, split=args.split, few_shot_min_set=args.use_min_cover, num_few_shot_examples=args.num_few_shot_examples)
    elif args.dataset == "colours":
        rules_file = "./data/colours/colours.csv"
        task = _task(rules_file, prompt_style=args.prompt_type, few_shot_min_set=args.use_min_cover, num_few_shot_examples=args.num_few_shot_examples)
    
    return task

def do_task(task, model_name, prompt_type, temp, start_ind: int = 0, end_ind: int = None, get_grammar_only: bool = False, get_completion_fn: Callable = get_completion_openai):
    correct = 0
    results_log = {"inputs": [], "outputs": [], "answer": [], "correct": []}
    system_prompt = task.get_system_prompt()
    total_prompt_tokens = 0
    total_completion_tokens = 0
    end_ind = end_ind if end_ind is not None else len(task)

    for i in tqdm(range(len(task))):
        if i < start_ind:
            continue
        if end_ind is not None and i >= end_ind:
            break
        
        if prompt_type == "base":
            input_prompt = task.get_standard_prompt(i)
        else:
            # to save on costs after grammars are already induced, just use a cached one.
            input_prompt, input_completion_num_tokens, input_prompt_num_tokens = task.get_special_prompt(i, return_grammar_only=get_grammar_only, use_cached=not get_grammar_only)

            if get_grammar_only:
                return None, input_prompt, input_completion_num_tokens, input_prompt_num_tokens

        message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input_prompt},
        ]

        output = get_completion_fn(message, model_name, temp)

        answer = task.get_answer(i)
        
        if "gpt" not in model_name: # open source models
            output_text = output[0].text # don't need to count tokens for cost
        else:
            completion_num_tokens = output["usage"]["completion_tokens"]
            prompt_num_tokens = output["usage"]["prompt_tokens"]
            output_text = output["choices"][0]["message"]["content"]
            total_completion_tokens += completion_num_tokens
            total_prompt_tokens += prompt_num_tokens
        
        results_log["inputs"].append(input_prompt)
        results_log["outputs"].append(output_text)
        is_correct = task.validate(i, output_text)
        results_log["answer"].append(answer)
        results_log["correct"].append(is_correct)
        correct += is_correct
    
    return correct/(end_ind - start_ind), results_log, total_completion_tokens, total_prompt_tokens

def finish_task(args: argparse.Namespace, acc: float, results_log: dict, output_file: str, get_grammar_only: bool = False) -> None:
    if not get_grammar_only:
        print(f"Accuracy: {acc}")
        results_df = pd.DataFrame(results_log)
        results_df.to_csv(output_file, index=False)
    else:
        info = {
            "temp": args.temp,
            "model": args.model,
            "date": datetime.datetime.now().strftime("%Y-%m-%d"),
            "grammar": results_log,
            "human_validated": False,
            "is_correct": None,
            "human_corrected": False
        } 
        mode = "w" if not os.path.exists(f"./data/{args.dataset}/gpt_4_induced_grammars.jsonl") else "a"
        with jsonlines.open(f"./data/{args.dataset}/gpt_4_induced_grammars.jsonl", mode) as writer:
            writer.write(info)

# from the tree of thoughts repo.
def gpt_usage(completion_tokens: int, prompt_tokens: int, backend="gpt-4"):
    if backend == "gpt-4":
        cost = completion_tokens / 1000 * 0.06 + prompt_tokens / 1000 * 0.03
    elif backend == "gpt-3.5-turbo":
        cost = (completion_tokens + prompt_tokens) / 1000 * 0.0002
    return {"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens, "cost": cost}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prompt OpenAI models with task specs")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", choices=["gpt-4", "gpt-3.5-turbo"], help="OpenAI model to use")
    parser.add_argument("--temp", default=0.0, type=float, help="Temperature for sampling")
    parser.add_argument("--dataset", required=True, choices=["scan", "cogs", "colours"])
    parser.add_argument("--split", default="simple", choices=["simple", "length", "jump", "cp_recursion", "prim_to_subj_common", "exposure_example_obj_proper", "obj_to_subj_common", "only_seen_as_unacc_subj_as_obj_omitted_transitive_subj"])
    parser.add_argument("--prompt_type", default="base", choices=["base", "full_grammar", "grammar_induction", "rule_selection"])
    parser.add_argument("--output", type=str)
    parser.add_argument("--start_ind", type=int)
    parser.add_argument("--end_ind", type=int)
    parser.add_argument("--num_few_shot_examples", type=int, default=5)
    parser.add_argument("--use_min_cover", action="store_true")
    parser.add_argument("--return_induced_grammar_only", action="store_true")

    args = parser.parse_args()

    if args.return_induced_grammar_only and not args.prompt_type == "grammar_induction":
        raise ValueError("Can only return induced grammar if prompt type is grammar_induction")
    
    output_file = args.output if args.output is not None else f"./logs/{args.dataset}_{args.split}_{args.prompt_type}_{args.model}_{args.start_ind}_{args.end_ind}_minset_{args.use_min_cover}.csv"
    openai.api_key = os.environ["OPENAI_API_KEY"]
    try:
        task = init_task(args)
        start_ind = args.start_ind if args.start_ind is not None else 0
        end_ind = args.end_ind if args.end_ind is not None else len(task)
        acc, results_log, total_completion_tokens, total_prompt_tokens = do_task(task, args.model, args.prompt_type, args.temp, start_ind=start_ind, end_ind=end_ind, get_grammar_only=args.return_induced_grammar_only)
        finish_task(args, acc, results_log, output_file, get_grammar_only=args.return_induced_grammar_only)
        cost = gpt_usage(total_completion_tokens, total_prompt_tokens, backend=args.model)
        print(f"Cost: {cost}")
    except ValueError:
        raise ValueError(f"Dataset {args.dataset} not registered")
    


    

