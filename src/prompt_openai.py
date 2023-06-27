import argparse
import openai
import os
from tqdm import tqdm
import backoff
from typing import List
import pandas as pd

from src import get_task

def do_task(task, model_name, prompt_type, temp):
    correct = 0
    results_log = {"inputs": [], "outputs": [], "correct": []}
    for i in tqdm(range(len(task))):
        if prompt_type == "base":
            input_prompt = task.get_standard_prompt(i)
        else:
            input_prompt = task.get_special_prompt(i)
        system_prompt = task.get_system_prompt(i)
        message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input_prompt},
        ]
        output = get_completion(message, model_name, temp)
        output_text = output["choices"][0]["message"]["content"]

        results_log["inputs"].append(input_prompt)
        results_log["outputs"].append(output_text)
        is_correct = task.validate(i, output)
        results_log["correct"].append(is_correct)
        correct += is_correct
        if i > 20:
            break
    
    return correct/20, results_log

def finish_task(acc: float, results_log: dict, output_file: str) -> None:
    print(f"Accuracy: {acc}")
    results_df = pd.DataFrame(results_log)
    results_df.to_csv(output_file, index=False)

def backoff_printer(details):
    print(f"Backing off {details['wait']} seconds after {details['tries']} tries calling function {details['target'].__name__} with args {details['args']} and kwargs {details['kwargs']}")
    
@backoff.on_exception(backoff.expo, openai.error.APIError, max_tries=30, on_backoff=backoff_printer)
def get_completion(prompts: List, model_name: str, temp: float = 0.7) -> str:
    completion = openai.ChatCompletion.create(
        model=model_name,
        messages=prompts,
        temperature=temp,
    )
    return completion
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prompt OpenAI models with task specs")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", choices=["gpt-4", "gpt-3.5-turbo", "text-davinci-003"], help="OpenAI model to use")
    parser.add_argument("--temp", default=0.0, type=float, help="Temperature for sampling")
    parser.add_argument("--dataset", default="scan", choices=["scan"])
    parser.add_argument("--split", default="simple", choices=["simple", "length", "jump"])
    parser.add_argument("--prompt_type", default="base", choices=["base", "full_grammar", "grammar_induction"])
    parser.add_argument("--output", type=str)

    args = parser.parse_args()
    
    output_file = args.output if args.output is not None else f"./logs/{args.dataset}_{args.split}_{args.prompt_type}_{args.model}.csv"
    openai.api_key = os.environ["OPENAI_API_KEY"]
    try:
        _task = get_task(args.dataset)
        if args.split == "simple":
            train_file = "./data/scan_simple_train.csv"
            test_file = "./data/scan_simple_test.csv"
        elif args.split == "length":
            train_file = "./data/scan_length_train.csv"
            test_file = "./data/scan_length_test.csv"
        elif args.split == "jump":
            train_file = "./data/scan_jump_train.csv"
            test_file = "./data/scan_jump_test.csv"
        else:
            raise ValueError(f"Split {args.split} not registered")
        
        task = _task(train_file, test_file, prompt_style=args.prompt_type)
        acc, results_log = do_task(task, args.model, args.prompt_type, args.temp)
        finish_task(acc, results_log, output_file)

    except ValueError:
        raise ValueError(f"Dataset {args.dataset} not registered")
    


    

