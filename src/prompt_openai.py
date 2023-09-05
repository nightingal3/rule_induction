import argparse
import openai
import os
from tqdm import tqdm
import backoff
from typing import List, Callable, Tuple
import pandas as pd
import jsonlines
import datetime
import pprint
from collections import defaultdict
import logging

from src import get_task

def backoff_printer(details):
    logging.info(f"Backing off {details['wait']} seconds after {details['tries']} tries calling function {details['target'].__name__} with args {details['args']} and kwargs {details['kwargs']}")
    
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
        task = _task(train_file, test_file, prompt_style=args.prompt_type, split=args.split, few_shot_min_set=args.use_min_cover, num_few_shot_examples=args.num_few_shot_examples, grammar_induction_loop=args.prompt_in_loop and args.prompt_type == "grammar_induction")
    elif args.dataset == "cogs":
        train_file = "./data/cogs/train_100.tsv"
        test_file = "./data/cogs/gen.tsv"
        task = _task(train_file, test_file, prompt_style=args.prompt_type, split=args.split, few_shot_min_set=args.use_min_cover, num_few_shot_examples=args.num_few_shot_examples)
    elif args.dataset == "colours":
        rules_file = "./data/colours/colours.csv"
        task = _task(rules_file, prompt_style=args.prompt_type, few_shot_min_set=args.use_min_cover, num_few_shot_examples=args.num_few_shot_examples, grammar_induction_loop=args.prompt_in_loop and args.prompt_type == "grammar_induction")
    elif args.dataset == "arc":
        task = _task(split=args.split, prompt_style=args.prompt_type, few_shot_min_set=args.use_min_cover, num_few_shot_examples=args.num_few_shot_examples, grammar_induction_loop=args.prompt_in_loop and args.prompt_type == "grammar_induction")
    elif args.dataset == "cherokee":
        if args.split == "simple":
            train_file = "./data/cherokee/cherokee_simple_train.csv"
            test_file = "./data/cherokee/cherokee_simple_test.csv"
            dev_file = "./data/cherokee/cherokee_simple_dev.csv"
        elif args.split == "ood":
            train_file = "./data/cherokee/cherokee_ood_train.csv"
            test_file = "./data/cherokee/cherokee_ood_test.csv"
            dev_file = "./data/cherokee/cherokee_ood_dev.csv"
        elif args.split == "easy":
            train_file = "./data/cherokee/cherokee_easy_train.csv"
            test_file = "./data/cherokee/cherokee_easy_test.csv"
            dev_file = "./data/cherokee/cherokee_easy_dev.csv"
        elif args.split == "debug":
            train_file = "./data/cherokee/cherokee_easy_train.csv"
            test_file = "./data/cherokee/cherokee_easy_test.csv"
            dev_file = "./data/cherokee/cherokee_easy_dev.csv"

        task = _task(train_file, test_file, dev_file, prompt_style=args.prompt_type, split=args.split, few_shot_min_set=args.use_min_cover, num_few_shot_examples=args.num_few_shot_examples, tgt_lang="en", grammar_induction_loop=args.prompt_in_loop and args.prompt_type == "grammar_induction", use_dev=True, dictionary_file="./data/cherokee-panlex/translations.csv")
    elif args.dataset == "naclo":
        task = _task(prompt_style=args.prompt_type, num_few_shot_examples=args.num_few_shot_examples, grammar_induction_loop=args.prompt_in_loop and args.prompt_type == "grammar_induction")

    return task

def get_completion_and_validate() -> Tuple[str, bool]:
    pass

def do_task(task, model_name, prompt_type, temp, start_ind: int = 0, end_ind: int = None, get_grammar_only: bool = False, get_completion_fn: Callable = get_completion_openai, use_test_set: bool = False):
    correct = 0 # note: this is accuracy for I/O tasks, but some MT metric for MT tasks.
    score_bleu = 0
    score_rouge = 0
    score_meteor = 0
    score_bertscore = 0

    results_log = defaultdict(list, {"inputs": [], "outputs": [], "answer": [], "correct": []})

    system_prompt = task.get_system_prompt()
    total_prompt_tokens = 0
    total_completion_tokens = 0
    end_ind = end_ind if end_ind is not None else len(task)
    num_not_skipped = 0

    for i in tqdm(range(len(task))):
        if i < start_ind:
            continue
        if end_ind is not None and i >= end_ind:
            break

        if prompt_type == "base":
            input_prompt, *rest = task.get_standard_prompt(i)
        else:
            # to save on costs after grammars are already induced, just use a cached one.
            use_cached = not get_grammar_only and not task.task_type == "arc"
            input_prompt, input_completion_num_tokens, input_prompt_num_tokens, *rest = task.get_special_prompt(i, return_grammar_only=get_grammar_only, use_cached=use_cached)

            if get_grammar_only:
                return None, input_prompt, input_completion_num_tokens, input_prompt_num_tokens
        #task.process_en_svo()
        #import pdb; pdb.set_trace()
        if len(rest) > 0 and len(rest[0]) > 0:
            # ARC/naclo
            task_id = rest[0]
            results_log["task_id"].append(task_id)

        message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input_prompt},
        ]

        # Rough
        if len(input_prompt.split()) > 4000:
            logging.info(f"Skipping input prompt on index {i} because it's too long.")
            results_log["inputs"].append(input_prompt)
            results_log["outputs"].append("skipped")
            results_log["answer"].append(task.get_answer(i))
            results_log["correct"].append(None)
            continue

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
        num_not_skipped += 1

        # TODO: arc allows 3 retries. Retry twice else fail.
            
        # mt has multiple metrics
        if task.task_type == "translation":
            if "exact_match" in is_correct:
                logging.debug("Output text: ", output_text)
                correct += is_correct["exact_match"]
                results_log["is_correct"].append(is_correct["exact_match"])
                
                logging.debug(f"Correct: {correct}")
            score_bleu += is_correct["bleu"]["bleu"]
            score_rouge += is_correct["rouge"]["rougeL"]
            score_meteor += is_correct["meteor"]["meteor"]
            if "bertscore" in is_correct:
                score_bertscore += is_correct["bertscore"]["f1"][0]
            results_log["bleu"].append(is_correct["bleu"]["bleu"])
            results_log["rouge"].append(is_correct["rouge"]["rougeL"])
            results_log["meteor"].append(is_correct["meteor"]["meteor"])
            if "bertscore" in is_correct:
                results_log["bertscore"].append(is_correct["bertscore"]["f1"][0])
            
        else:
            results_log["correct"].append(is_correct)
            correct += is_correct
            logging.debug(f"Correct: {correct}")
        
        results_log["answer"].append(answer)

    if task.task_type == "translation" and correct == 0:
        correct = score_bleu

    return correct/num_not_skipped, results_log, total_completion_tokens, total_prompt_tokens

def finish_task(args: argparse.Namespace, acc: float, results_log: dict, output_file: str, get_grammar_only: bool = False) -> None:
    if not get_grammar_only:
        logging.info(f"Accuracy: {acc}")
        if "bleu" in results_log:
            logging.info(f"BLEU: {sum(results_log['bleu'])/len(results_log['bleu'])}")
            logging.info(f"ROUGE: {sum(results_log['rouge'])/len(results_log['rouge'])}")
            logging.info(f"Meteor: {sum(results_log['meteor'])/len(results_log['meteor'])}")
            if len(results_log["bertscore"]) > 0:
                logging.info(f"BertScore: {sum(results_log['bertscore'])/len(results_log['bertscore'])}")
        
        results_df = pd.DataFrame({k: v for k, v in results_log.items() if len(v) > 0})
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
    parser.add_argument("--dataset", required=True, choices=["scan", "cogs", "colours", "cherokee", "arc", "naclo"])
    parser.add_argument("--split", default="simple", choices=["simple", "easy", "length", "jump", "cp_recursion", "prim_to_subj_common", "exposure_example_obj_proper", "obj_to_subj_common", "only_seen_as_unacc_subj_as_obj_omitted_transitive_subj", "debug", "AboveBelow", "CleanUp"])
    parser.add_argument("--prompt_type", default="base", choices=["base", "full_grammar", "grammar_induction", "rule_selection", "vocab_induction"])
    parser.add_argument("--output", type=str)
    parser.add_argument("--start_ind", type=int)
    parser.add_argument("--end_ind", type=int)
    parser.add_argument("--num_few_shot_examples", type=int, default=5)
    parser.add_argument("--use_min_cover", action="store_true", help="Use a curated set of few-shot examples that contain all primitives")
    parser.add_argument("--return_induced_grammar_only", action="store_true")
    parser.add_argument("--prompt_in_loop", help="Only for grammar induction. Present a few examples at a time until rules converge.", action="store_true")
    parser.add_argument("--debug", action="store_true", help="Show debug info (prompts and num correct)")

    args = parser.parse_args()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO if not args.debug else logging.DEBUG)

    if args.return_induced_grammar_only and not args.prompt_type == "grammar_induction":
        raise ValueError("Can only return induced grammar if prompt type is grammar_induction")
    
    output_file = args.output if args.output is not None else f"./logs/{args.dataset}_{args.split}_{args.prompt_type}_{args.model}_{args.start_ind}_{args.end_ind}_minset_{args.use_min_cover}_loop_{args.prompt_in_loop}.csv"
    openai.api_key = os.environ["OPENAI_API_KEY"]
    try:
        task = init_task(args)
        start_ind = args.start_ind if args.start_ind is not None else 0
        end_ind = args.end_ind if args.end_ind is not None else len(task)
        acc, results_log, total_completion_tokens, total_prompt_tokens = do_task(task, args.model, args.prompt_type, args.temp, start_ind=start_ind, end_ind=end_ind, get_grammar_only=args.return_induced_grammar_only)
        finish_task(args, acc, results_log, output_file, get_grammar_only=args.return_induced_grammar_only)
        cost = gpt_usage(total_completion_tokens, total_prompt_tokens, backend=args.model)
        logging.info(f"Cost: {cost}")
    except ValueError:
        raise ValueError(f"Dataset {args.dataset} not registered")
    


    

