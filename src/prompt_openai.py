import argparse
import openai
import os
from tqdm import tqdm
import backoff
from typing import List, Callable, Tuple, Optional, Union
import pandas as pd
import jsonlines
import datetime
import pprint
from collections import defaultdict
import logging
import pathlib
import signal
import sys

from src import get_task


llama_wrapper = """<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>

{user_prompt} [/INST]"""  # TODO: not sure if we need this, the generation seems to be empty with this thing


all_error_types = (
    openai.error.OpenAIError,
    openai.error.APIError,
    openai.error.RateLimitError,
    openai.error.APIConnectionError,
    openai.error.ServiceUnavailableError,
)

# These are global to handle sigint/sigterm in case of openai issues
correct = 0  # note: this is accuracy for I/O tasks, but some MT metric for MT tasks.
score_bleu = 0
score_chrf = 0
score_meteor = 0
score_bertscore = 0

results_log = defaultdict(
    list, {"inputs": [], "outputs": [], "answer": [], "correct": []}
)
proposed_hypotheses = defaultdict(
    list, {"all_hyps": [], "all_probs": [], "task_id": []}
)

total_prompt_tokens = 0
total_completion_tokens = 0
num_not_skipped = 0
total_processed = 0

start_ind = None
end_ind = None


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
) -> str:
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
    else:
        completion = openai.ChatCompletion.create(
            model=model_name,
            messages=prompts,
            temperature=temp,
            logprobs=logprobs,
            max_tokens=max_tokens,
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
        task = _task(
            train_file,
            test_file,
            prompt_style=args.prompt_type,
            split=args.split,
            few_shot_min_set=args.use_min_cover,
            num_few_shot_examples=args.num_few_shot_examples,
            grammar_induction_loop=args.prompt_in_loop
            and args.prompt_type == "grammar_induction",
        )
    elif args.dataset == "cogs":
        train_file = "./data/cogs/train_100.tsv"
        test_file = "./data/cogs/gen.tsv"
        task = _task(
            train_file,
            test_file,
            prompt_style=args.prompt_type,
            split=args.split,
            few_shot_min_set=args.use_min_cover,
            num_few_shot_examples=args.num_few_shot_examples,
        )
    elif args.dataset == "colours":
        rules_file = "./data/colours/colours.csv"
        task = _task(
            rules_file,
            prompt_style=args.prompt_type,
            few_shot_min_set=args.use_min_cover,
            num_few_shot_examples=args.num_few_shot_examples,
            grammar_induction_loop=args.prompt_in_loop
            and args.prompt_type == "grammar_induction",
        )
    elif args.dataset == "arc":
        task = _task(
            split=args.split,
            prompt_style=args.prompt_type,
            few_shot_min_set=args.use_min_cover,
            num_few_shot_examples=args.num_few_shot_examples,
            grammar_induction_loop=args.prompt_in_loop
            and args.prompt_type == "grammar_induction",
        )
    elif args.dataset == "functions":
        task = _task(
            prompt_style=args.prompt_type,
            num_few_shot_examples=args.num_few_shot_examples,
            degree=args.degree,
        )
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

        task = _task(
            train_file,
            test_file,
            dev_file,
            prompt_style=args.prompt_type,
            split=args.split,
            few_shot_min_set=args.use_min_cover,
            num_few_shot_examples=args.num_few_shot_examples,
            tgt_lang="en",
            grammar_induction_loop=args.prompt_in_loop
            and args.prompt_type == "grammar_induction",
            use_dev=True,
            dictionary_file="./data/cherokee-panlex/translations.csv",
        )
    elif args.dataset == "naclo":
        task = _task(
            prompt_style=args.prompt_type,
            num_few_shot_examples=args.num_few_shot_examples,
            grammar_induction_loop=args.prompt_in_loop
            and args.prompt_type == "grammar_induction",
        )

    return task


def get_completion_and_validate() -> Tuple[str, bool]:
    pass


def do_task(
    task,
    model_name,
    prompt_type,
    temp,
    start_ind: int = 0,
    end_ind: int = None,
    get_grammar_only: bool = False,
    get_completion_fn: Callable = get_completion_openai,
    use_test_set: bool = False,
    rejection_sampling: bool = False,
    no_few_shot_examples: bool = False,
    hyp_reranking_method: str = "ground_truth",
    num_hyps: int = 1,
):
    global correct
    global score_bleu
    global score_chrf
    global score_meteor
    global score_bertscore

    global results_log
    global proposed_hypotheses

    global total_prompt_tokens
    global total_completion_tokens
    global num_not_skipped
    global total_processed

    system_prompt = task.get_system_prompt()

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
            # input_prompt, input_completion_num_tokens, input_prompt_num_tokens, *rest = task.get_special_prompt(i, return_grammar_only=get_grammar_only, use_cached=use_cached, no_few_shot_examples=args.no_few_shot_examples)
            (
                input_prompt,
                input_completion_num_tokens,
                input_prompt_num_tokens,
                *rest,
            ) = task.get_special_prompt(
                i,
                return_grammar_only=get_grammar_only,
                no_few_shot_examples=no_few_shot_examples,
                backend=model_name,
                n_hyps=num_hyps,
                rerank_by=hyp_reranking_method,
            )
            # TODO: always have backend be GPT-4, or use the same model?

            if get_grammar_only:
                results_log["outputs"].append(input_prompt)
                results_log["inputs"].append(task.get_standard_prompt(i))
                if args.rejection_sampling:
                    results_log["num_times_changed"].append(rest[0])
                num_not_skipped += 1
                continue

        # task.process_en_svo()
        # breakpoint()
        if len(rest) > 0 and len(rest[0]) > 0:
            # ARC/naclo
            task_id = rest[0]
            results_log["task_id"].append(task_id)
            # TODO - this is a bug
            # add alternate hypotheses and probabilities to the results log if they exist in the task
            if (
                hasattr(task, "proposed_hypotheses")
                and len(task.proposed_hypotheses["hypothesis"]) > 0
            ):
                proposed_hypotheses["all_hyps"].extend(
                    task.proposed_hypotheses["hypothesis"][-1]
                )
                proposed_hypotheses["all_probs"].extend(
                    task.proposed_hypotheses["estimated_prob"][-1]
                )
                proposed_hypotheses["task_id"].extend(
                    [task_id] * len(task.proposed_hypotheses["hypothesis"][-1])
                )

        message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input_prompt},
        ]

        # if "gpt" not in model_name: # open source models
        # message = llama_wrapper.format(system_prompt=system_prompt, user_prompt=input_prompt)

        # Rough
        total_processed += 1

        if len(input_prompt.split()) > 4000:
            logging.info(f"Skipping input prompt on index {i} because it's too long.")
            results_log["inputs"].append(input_prompt)
            results_log["outputs"].append("skipped")
            results_log["answer"].append(task.get_answer(i))
            results_log["correct"].append(None)
            continue

        output = get_completion_fn(message, model_name=model_name, temp=temp)
        answer = task.get_answer(i)

        if "gpt" not in model_name:  # open source models
            output_text = output  # don't need to count tokens for cost

        else:
            completion_num_tokens = output["usage"]["completion_tokens"]
            prompt_num_tokens = output["usage"]["prompt_tokens"]
            output_text = output["choices"][0]["message"]["content"]
            total_completion_tokens += completion_num_tokens
            total_prompt_tokens += prompt_num_tokens

        results_log["inputs"].append(input_prompt)
        results_log["outputs"].append(output_text)
        if task.task_type == "translation":
            _, ref, hyp = task.validate(i, output_text, return_ref_and_hyp_only=True)
            results_log["ref"].append(ref)
            results_log["hyp"].append(hyp)
        else:
            is_correct = task.validate(i, output_text)
        num_not_skipped += 1

        # TODO: arc allows 3 retries. Retry twice else fail.

        if task.task_type != "translation":
            results_log["correct"].append(is_correct)
            correct += is_correct
            logging.info(f"Correct: {correct}")

        results_log["answer"].append(answer)

    if task.task_type == "translation":
        scores = task.validate_all(results_log["ref"], results_log["hyp"])
        results_log["bleu"] = [scores["sacrebleu"]["score"]] * len(results_log["ref"])
        results_log["chrf"] = [scores["chrf"]["score"]] * len(results_log["ref"])
        results_log["meteor"] = [scores["meteor"]["meteor"]] * len(results_log["ref"])
        if "bertscore" in scores:
            results_log["bertscore"] = scores["bertscore"]["f1"][0]

    return (
        correct / num_not_skipped,
        results_log,
        total_completion_tokens,
        total_prompt_tokens,
    )


def make_finish_task(
    args_for_task: argparse.Namespace,
    output_file_orig: str,
    get_grammar_only: bool = False,
    start_ind: int = 0,
) -> Callable:
    def finish_task(*args, **kwargs) -> None:
        global correct
        global num_not_skipped
        global total_processed
        global results_log
        global proposed_hypotheses

        nonlocal args_for_task
        nonlocal output_file_orig
        nonlocal get_grammar_only
        nonlocal start_ind

        output_file = output_file_orig
        acc = correct / num_not_skipped
        end_ind = start_ind + total_processed
        logging.info(f"Accuracy: {acc}")
        if "bleu" in results_log:
            logging.info(f"BLEU: {results_log['bleu']}")
            logging.info(f"chrF: {results_log['chrf']}")
            logging.info(f"Meteor: {results_log['meteor']}")
            if len(results_log["bertscore"]) > 0:
                logging.info(f"BertScore: {results_log['bertscore']}")

        results_df = pd.DataFrame(
            {k: v for k, v in results_log.items() if isinstance(v, list) and len(v) > 0}
        )

        # append true start and end inds to filename
        output_file = output_file.replace(
            ".csv", f"_start_{start_ind}_end_{end_ind}.csv"
        )

        if get_grammar_only:
            output_file = output_file.replace(".csv", "_induced_grammar.csv")
        if args_for_task.no_few_shot_examples:
            output_file = output_file.replace(".csv", "_no_few_shot_examples.csv")
        if args_for_task.dataset == "functions":
            output_file = output_file.replace(
                ".csv", f"_degree_{args_for_task.degree}.csv"
            )
        if args_for_task.num_hyps != 1:
            output_file = output_file.replace(
                ".csv",
                f"_num_hyps_{args_for_task.num_hyps}_{args_for_task.hyp_reranking_method}.csv",
            )
        if not os.path.exists(os.path.dirname(output_file)):
            pathlib.Path(os.path.dirname(output_file)).mkdir(
                parents=True, exist_ok=True
            )
        results_df.to_csv(output_file, index=False)
        logging.info("Wrote results to " + output_file)

        # if alternate hypotheses are stored, dump them to a separate file
        if len(proposed_hypotheses["all_hyps"]) > 0:
            hyps_file = output_file.replace(
                ".csv",
                f"_{args_for_task.hyp_reranking_method}_HYPSFILE.csv",
            )
            hyps_df = pd.DataFrame(proposed_hypotheses)
            hyps_df.to_csv(hyps_file, index=False)
            logging.info("Wrote hypotheses to " + hyps_file)

        sys.exit(0)

    return finish_task


# from the tree of thoughts repo.
def gpt_usage(backend="gpt-4"):
    global total_completion_tokens
    global total_prompt_tokens
    if backend == "gpt-4":
        cost = total_completion_tokens / 1000 * 0.06 + total_prompt_tokens / 1000 * 0.03
    elif backend == "gpt-3.5-turbo":
        cost = (total_completion_tokens + total_prompt_tokens) / 1000 * 0.0002
    return {
        "completion_tokens": total_completion_tokens,
        "prompt_tokens": total_prompt_tokens,
        "cost": cost,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prompt OpenAI models with task specs")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-3.5-turbo",
        choices=["gpt-4", "gpt-3.5-turbo", "gpt-4-turbo"],
        help="OpenAI model to use",
    )
    parser.add_argument(
        "--temp", default=0.0, type=float, help="Temperature for sampling"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["scan", "cogs", "colours", "cherokee", "arc", "naclo", "functions"],
    )
    parser.add_argument(
        "--split",
        default="simple",
        choices=[
            "simple",
            "easy",
            "length",
            "jump",
            "cp_recursion",
            "prim_to_subj_common",
            "exposure_example_obj_proper",
            "obj_to_subj_common",
            "only_seen_as_unacc_subj_as_obj_omitted_transitive_subj",
            "debug",
            "AboveBelow",
            "CleanUp",
        ],
    )
    parser.add_argument(
        "--prompt_type",
        default="base",
        choices=[
            "base",
            "full_grammar",
            "grammar_induction",
            "rule_selection",
            "vocab_induction",
        ],
    )
    parser.add_argument("--output", type=str)
    parser.add_argument("--start_ind", type=int)
    parser.add_argument("--end_ind", type=int)
    parser.add_argument("--num_few_shot_examples", type=int, default=5)
    parser.add_argument(
        "--use_min_cover",
        action="store_true",
        help="Use a curated set of few-shot examples that contain all primitives",
    )
    parser.add_argument("--return_induced_grammar_only", action="store_true")
    parser.add_argument(
        "--prompt_in_loop",
        help="Only for grammar induction. Present a few examples at a time until rules converge.",
        action="store_true",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Show debug info (prompts and num correct)"
    )
    parser.add_argument(
        "--no_few_shot_examples",
        action="store_true",
        help="Don't show few shot examples at all (for full_grammar and grammar_induction type prompts)",
    )
    parser.add_argument(
        "--rejection_sampling",
        action="store_true",
        help="Use rejection sampling to get a valid completion",
    )
    parser.add_argument(
        "--num_hyps",
        type=int,
        default=1,
        help="number of hypotheses to generate (for use with grammar_induction prompt type)",
    )
    parser.add_argument(
        "--hyp_reranking_method",
        type=str,
        choices=[
            "ground_truth",
            "p_data_given_hyp_guess",
            "p_data_given_hyp_logprobs",
            "p_answer_given_hyp_logprobs",
        ],
        default="ground_truth",
    )
    parser.add_argument(
        "--degree",
        help="polynomial degree for the functions domain",
        type=int,
        default=1,
    )

    args = parser.parse_args()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO if not args.debug else logging.DEBUG)

    # handle openai getting stuck: write current results to file
    start_ind = args.start_ind if args.start_ind is not None else 0
    output_file = (
        args.output
        if args.output is not None
        else f"./logs/{args.dataset}/{args.dataset}_{args.split}_{args.prompt_type}_{args.model}_minset_{args.use_min_cover}_loop_{args.prompt_in_loop}_temp_{args.temp}_few_shot_examples_{args.no_few_shot_examples}.csv"
    )

    finish_task = make_finish_task(
        args, output_file, args.return_induced_grammar_only, start_ind
    )
    signal.signal(signal.SIGINT, finish_task)
    signal.signal(signal.SIGTERM, finish_task)

    start_ind = args.start_ind if args.start_ind is not None else 0

    if args.return_induced_grammar_only and not args.prompt_type == "grammar_induction":
        raise ValueError(
            "Can only return induced grammar if prompt type is grammar_induction"
        )

    openai.api_key = os.environ["OPENAI_API_KEY"]
    model_fixed_versions = {
        "gpt-3.5-turbo": "gpt-3.5-turbo-0613",
        "gpt-4": "gpt-4-0314",
        "gpt-4-turbo": "gpt-4-1106-preview",
    }

    model_name = (
        model_fixed_versions[args.model]
        if args.model in model_fixed_versions
        else args.model
    )
    try:
        task = init_task(args)
        end_ind = args.end_ind if args.end_ind is not None else len(task)
        acc, results_log, total_completion_tokens, total_prompt_tokens = do_task(
            task,
            model_name,
            args.prompt_type,
            args.temp,
            start_ind=start_ind,
            end_ind=end_ind,
            get_grammar_only=args.return_induced_grammar_only,
            rejection_sampling=args.rejection_sampling,
            no_few_shot_examples=args.no_few_shot_examples,
            num_hyps=args.num_hyps,
            hyp_reranking_method=args.hyp_reranking_method,
        )
        finish_task(
            args,
            acc,
            results_log,
            output_file,
            get_grammar_only=args.return_induced_grammar_only,
            num_hyps=args.num_hyps,
            hyp_reranking_method=args.hyp_reranking_method,
        )
        cost = gpt_usage(
            total_completion_tokens, total_prompt_tokens, backend=args.model
        )
        logging.info(f"Cost: {cost}")
    except KeyboardInterrupt:
        pass
