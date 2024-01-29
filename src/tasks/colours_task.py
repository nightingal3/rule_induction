from typing import List, Literal, Tuple, Optional

import pandas as pd
import random
import json
import jsonlines
import numpy as np
import os
from collections import defaultdict
import re

from src.prompts.colours_prompts import *
from src.task import BaseTask
from src.get_completion import get_completion_openai
from src.utils.utils import try_convert_to_float
from src.utils.parsing_utils import (
    get_example_start_token,
    get_start_and_end_tokens,
    parse_production_rule,
    get_ground_truth_answers_vocab,
)


class ColoursTask(BaseTask):
    task_type = "io"

    def __init__(
        self,
        rules_file: str,
        prompt_style: Literal["base", "full_grammar", "grammar_induction", "zs-cot"],
        num_few_shot_examples: int = 5,
        nonce: bool = False,
        few_shot_min_set: bool = False,
        grammar_induction_loop: bool = False,
        test_data_file: Optional[str] = None,
        train_data_file: Optional[str] = None,
        write_data_dir: Optional[str] = None,
        split: str = "simple",
        **kwargs,
    ) -> None:
        self.rules = pd.read_csv(rules_file)

        self.write_data_dir = (
            write_data_dir if write_data_dir is not None else "./data/colours"
        )

        if test_data_file is not None and train_data_file is not None:
            self.train_data = pd.read_csv(train_data_file)
            self.test_data = pd.read_csv(test_data_file)
        else:
            self.train_data, self.test_data = self.generate_examples()

        self.prompt_style = prompt_style
        self.num_few_shot_examples = num_few_shot_examples
        self.is_nonce = nonce
        self.split = split
        # if prompt style is base, this will just select the min set examples.
        # if prompt style is full_grammar, this will use the example parses as well.
        self.use_few_shot_minset = few_shot_min_set
        if self.use_few_shot_minset:
            if self.split == "miniscan":
                self.min_examples = pd.read_csv(
                    "./data/colours/miniscan/train_data.csv"
                )
            else:
                self.min_examples = pd.read_csv(
                    "./data/colours/all_commands_minset.csv"
                )

        self.cached_induced_grammars = []
        if os.path.exists("./data/colours/gpt_4_induced_grammars.jsonl"):
            with jsonlines.open(
                "./data/colours/gpt_4_induced_grammars.jsonl", "r"
            ) as f:
                for line in f:
                    self.cached_induced_grammars.append(line)

        self.grammar_induction_loop = grammar_induction_loop
        self.src_vocab = list(self.rules["word"])
        self.proposed_hypotheses = {
            "hypothesis": [],
            "estimated_prob": [],
            "rank": [],
            "task_id": [],
            "for_word": [],
        }

    def generate_examples(self, data_size=1000) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if os.path.exists(f"{self.write_data_dir}/train_data.csv") and os.path.exists(
            f"{self.write_data_dir}/test_data.csv"
        ):
            train_data = pd.read_csv(f"{self.write_data_dir}/train_data.csv")
            test_data = pd.read_csv(f"{self.write_data_dir}/test_data.csv")
        else:
            colour_mappings = self.rules.loc[self.rules["rule_type"] == "colour"]
            number_mappings = self.rules.loc[self.rules["rule_type"] == "number"]
            concept_mappings = self.rules.loc[self.rules["rule_type"] == "concept"]
            # TODO: finish the concept mappings to round out the colours dataset
            all_examples = []
            prev_colour = None
            for _ in range(data_size):
                num_colours = np.random.choice(
                    np.arange(1, 6), p=[0.4, 0.3, 0.15, 0.1, 0.05]
                )
                command = ""
                meaning = ""
                for _ in range(num_colours):
                    colour = np.random.choice(colour_mappings["meaning"])
                    while (
                        colour == prev_colour
                    ):  # don't repeat so that number words are more meaningful
                        colour = np.random.choice(colour_mappings["meaning"])
                    command += colour_mappings.loc[
                        colour_mappings["meaning"] == colour
                    ]["word"].item()
                    num_repeats = np.random.choice(np.arange(1, 4), p=[0.8, 0.1, 0.1])
                    if num_repeats > 1:
                        command += (
                            " "
                            + number_mappings.loc[
                                number_mappings["meaning"] == str(num_repeats)
                            ]["word"].item()
                        )
                        meaning += " ".join([colour for _ in range(num_repeats)])
                    else:
                        meaning += colour
                    command += " "
                    meaning += " "
                    prev_colour = colour
                # delete last space
                command = command[:-1]
                meaning = meaning[:-1]
                all_examples.append((command, meaning))

            train_data = pd.DataFrame(
                all_examples[: int(data_size * 0.8)], columns=["commands", "actions"]
            )
            test_data = pd.DataFrame(
                all_examples[int(data_size * 0.8) :], columns=["commands", "actions"]
            )

            # save train and test data in data/colours
            train_data.to_csv("./data/colours/train_data.csv", index=False)
            test_data.to_csv("./data/colours/test_data.csv", index=False)

        return train_data, test_data

    def __len__(self) -> int:
        return len(self.test_data)

    def get_input(self, idx: int) -> str:
        return self.test_data.iloc[idx]["commands"]

    def get_answer(self, idx: int) -> str:
        return self.test_data.iloc[idx]["actions"]

    def get_few_shot_examples(
        self,
        idx: int,
        no_parse: bool = True,
        use_few_shot_minset: Optional[bool] = None,
        num_examples: Optional[int] = None,
        return_as_lst: bool = False,
    ) -> str:
        # get some random examples based on the idx (repeatable)
        use_minset = (
            self.use_few_shot_minset
            if use_few_shot_minset is None
            else use_few_shot_minset
        )
        if use_minset:
            if self.prompt_style == "base" or no_parse:
                few_shot_inputs = self.min_examples["commands"]
                few_shot_outputs = self.min_examples["actions"]
                if return_as_lst:
                    return [(i, o) for i, o in zip(few_shot_inputs, few_shot_outputs)]

                few_shot_formatted = self.few_shot_examples_wrap(
                    few_shot_inputs, few_shot_outputs
                )
            else:
                all_examples = self.min_examples["example_parse_few_shot"]
                if return_as_lst:
                    breakpoint()
                    return list(all_examples)
                few_shot_formatted = "\n\n".join(all_examples)
        else:
            num_examples_to_get = (
                self.num_few_shot_examples if num_examples is None else num_examples
            )
            random.seed(idx)
            indices = random.sample(range(len(self.train_data)), num_examples_to_get)
            few_shot_inputs = [self.train_data.iloc[i]["commands"] for i in indices]
            few_shot_outputs = [self.train_data.iloc[i]["actions"] for i in indices]
            if return_as_lst:
                return [(i, o) for i, o in zip(few_shot_inputs, few_shot_outputs)]
            few_shot_formatted = self.few_shot_examples_wrap(
                few_shot_inputs, few_shot_outputs
            )

        return few_shot_formatted

    def get_examples_with_one_word(self, word: str, num_examples: int = 5) -> List[str]:
        examples_src = []
        examples_tgt = []
        # shuffle train data
        self.train_data = self.train_data.sample(frac=1)
        for i, row in self.train_data.iterrows():
            if word in row["commands"].split(" "):
                examples_src.append(row["commands"])
                examples_tgt.append(row["actions"])
            if len(examples_src) >= num_examples:
                break
        return list(zip(examples_src, examples_tgt))

    def _validate(self, idx: int, output_text: str) -> bool:
        output_actions = output_text.split("Output: ")[-1].strip()
        return output_actions == self.test_data.iloc[idx]["actions"]

    def validate(
        self, idx: int, output_text: str, expected_answer: Optional[str] = None
    ):
        if expected_answer is None:
            answer = self.test_examples[idx]["output"]
        else:
            answer = expected_answer

        # Normalize line breaks and lowercase
        output_text = output_text.replace("\r\n", "\n").lower()

        # Pattern to find "Output:" followed by space-separated words, stopping at newline
        pattern = r"output:\s*([^\n]+)(?=\n|$)"
        matches = list(re.finditer(pattern, output_text, re.IGNORECASE | re.DOTALL))
        valid_output = None
        for match in matches:
            # Extract the context for this "Output:"
            start_pos = match.start()
            context = output_text[:start_pos]
            context_lines = context.split("\n")

            # Check if the context does not contain "Input:" immediately before this "Output:"
            if len(context_lines) < 2 or "input:" not in context_lines[-2]:
                valid_output = match.group(1).strip()

        return valid_output == answer

    def get_standard_prompt(self, idx: int) -> str:
        input = self.get_input(idx)
        few_shot_examples = self.get_few_shot_examples(idx)
        return self.standard_prompt_wrap(few_shot_examples, input), str(idx)

    def get_special_prompt(
        self,
        idx: int,
        backend: str = "gpt-4",
        return_grammar_only: bool = False,
        n_hyps: int = 1,
        use_cached: bool = False,
        rejection_sampling: bool = False,
        rerank_by: str = "p_data_given_hyp_guess",
        induce_one_word_at_a_time: bool = True,
        **kwargs,
    ) -> Tuple[str, int, int]:
        # TODO: implement ground truth reranking!!!
        if self.prompt_style == "full_grammar":
            return self.get_full_grammar_prompt(idx), 0, 0
        elif self.prompt_style == "zs-cot":
            return (self.get_zs_cot_prompt(idx), 0, 0)
        else:
            if self.prompt_style == "grammar_induction":
                few_shot_examples = self.get_few_shot_examples(
                    idx,
                    return_as_lst=True,
                    no_parse=True,
                    use_few_shot_minset=self.use_few_shot_minset,
                )
                few_shot_examples_str = "\n".join(
                    [
                        few_shot_examples_prompt.format(input=i, output=o)
                        for i, o in few_shot_examples
                    ]
                )

                if use_cached:
                    induced_grammar = self.cached_induced_grammars[0]["grammar"]
                    usage_completion = 0
                    usage_prompt = 0
                else:
                    (
                        induced_rules,
                        usage_completion,
                        usage_prompt,
                    ) = self.induce_word_by_word(
                        idx, self.src_vocab, backend, rerank_by=rerank_by, n_hyps=n_hyps
                    )
                    all_induced_rules = "\n".join(induced_rules.values())
                    induced_grammar = all_induced_rules_wrapped.format(
                        all_induced_rules=all_induced_rules
                    )

                    # grammar_induction_prompt = self.get_grammar_induction_prompt(
                    #     idx, no_parse=True
                    # )
                    # if "gpt" in backend:
                    #     message = [
                    #         {"role": "system", "content": GRAMMAR_INDUCTION_SYSPROMPT},
                    #         {"role": "user", "content": grammar_induction_prompt},
                    #     ]
                    #     if not rejection_sampling:
                    #         completion = self.get_best_grammar(
                    #             message,
                    #             backend,
                    #             n_hyps=n_hyps,
                    #             rerank_by=rerank_by,
                    #             few_shot_examples=few_shot_examples,
                    #         )
                    #     else:
                    #         completion = get_completion_openai(
                    #             message, backend, temp=1.0
                    #         )

                    #     induced_grammar = completion["choices"][0]["message"]["content"]

                    #     converged = False
                    #     num_times_changed = 0
                    #     while (self.grammar_induction_loop and not converged) or (
                    #         rejection_sampling and not converged
                    #     ):
                    #         repeat_prompt = self.get_repeat_prompt(
                    #             induced_grammar,
                    #             schedule="target_hard_words",
                    #             num_examples_to_get=5,
                    #         )
                    #         if self.grammar_induction_loop:
                    #             if num_times_changed == 0:
                    #                 message = [
                    #                     {
                    #                         "role": "system",
                    #                         "content": GRAMMAR_INDUCTION_SYSPROMPT,
                    #                     },
                    #                     {"role": "user", "content": repeat_prompt},
                    #                 ]
                    #             else:
                    #                 new_examples = self.get_repeat_prompt(
                    #                     induced_grammar,
                    #                     schedule="target_hard_words",
                    #                     num_examples_to_get=5,
                    #                     return_examples_only=True,
                    #                 )
                    #                 new_examples_str = "\n\n".join(new_examples)
                    #                 # if there are too many previous revisions (> 5, truncate to 5)
                    #                 if len(message) >= 5:
                    #                     message = message[-4:]
                    #                     message.insert(
                    #                         0,
                    #                         {
                    #                             "role": "system",
                    #                             "content": GRAMMAR_INDUCTION_SYSPROMPT,
                    #                         },
                    #                     )
                    #                 message.append(
                    #                     {
                    #                         "role": "user",
                    #                         "content": f"That wasn't quite right. Please observe these new examples and try again.\n\nNew Examples:\n"
                    #                         + new_examples_str
                    #                         + f"\nYour previous attempt:\n{induced_grammar}\n\nNew grammar:",
                    #                     }
                    #                 )

                    #             completion = get_completion_openai(
                    #                 message, backend, temp=1
                    #             )
                    #             induced_grammar_new = completion["choices"][0][
                    #                 "message"
                    #             ]["content"]
                    #             if "no changes" in induced_grammar_new.lower():
                    #                 converged = True
                    #             else:
                    #                 induced_grammar = induced_grammar_new
                    #                 num_times_changed += 1
                    #         elif rejection_sampling:
                    #             new_examples = self.get_repeat_prompt(
                    #                 induced_grammar,
                    #                 schedule="target_hard_words",
                    #                 num_examples_to_get=5,
                    #                 return_examples_only=True,
                    #             )
                    #             rules_split = induced_grammar.split("\n")
                    #             rules_dict = defaultdict(str)
                    #             for rule in rules_split:
                    #                 if "->" not in rule:
                    #                     continue
                    #                 try:
                    #                     lhs, rhs = rule.split("->")[:2]
                    #                 except:
                    #                     print("unparseable rule: ", rule)
                    #                     continue
                    #                 lhs = lhs.strip()
                    #                 rhs = rhs.strip()
                    #                 # remove numbering such as 1. or 2. or 3.
                    #                 lhs = lhs.split(". ")[-1]
                    #                 # remove left/right brackets

                    #                 if lhs not in rules_dict:
                    #                     rules_dict[lhs] = rhs

                    #             # use string substitution to test the rules. If the rules are incorrect, generate new rules.
                    #             all_correct = False
                    #             for example in new_examples:
                    #                 inp, oup = example.split("\n")
                    #                 inp = inp.split(": ")[1].strip()
                    #                 oup = oup.split(": ")[1].strip()

                    #                 inp = inp.split(". ")[-1]
                    #                 output_try = inp

                    #                 # replace all the lhs with rhs
                    #                 for lhs, rhs in rules_dict.items():
                    #                     if "bluf" in lhs or "walm" in lhs:
                    #                         continue  # repeat rules, do them last
                    #                     output_try = output_try.replace(lhs, rhs)

                    #                 # repeat rules
                    #                 bluf_rule = rules_dict["bluf"]
                    #                 walm_rule = rules_dict["walm"]
                    #                 keywords_bluf = [
                    #                     "repeat",
                    #                     "twice",
                    #                     "2",
                    #                     "two",
                    #                     "double",
                    #                     "before",
                    #                 ]
                    #                 keywords_walm = [
                    #                     "repeat",
                    #                     "thrice",
                    #                     "3",
                    #                     "three",
                    #                     "triple",
                    #                     "before",
                    #                 ]
                    #                 # if none of the keywords are present, it's probably wrong
                    #                 if not any(
                    #                     [
                    #                         keyword in bluf_rule
                    #                         for keyword in keywords_bluf
                    #                     ]
                    #                 ) or not any(
                    #                     [
                    #                         keyword in walm_rule
                    #                         for keyword in keywords_walm
                    #                     ]
                    #                 ):
                    #                     all_correct = False
                    #                     break

                    #                 # TODO: we can replace them but it's probably correct if it has the keywords
                    #                 split_output = output_try.split(" ")
                    #                 output_try_new = []
                    #                 # repeat the previous thing once more, or twice more
                    #                 for i, word in enumerate(split_output):
                    #                     if word != "bluf" and word != "walm":
                    #                         new_input.append(word)
                    #                     if word == "bluf":
                    #                         translated_word = rules_dict[
                    #                             split_output[i - 1]
                    #                         ]
                    #                         new_input.append(translated_word)
                    #                     elif word == "walm":
                    #                         translated_word = rules_dict[
                    #                             split_output[i - 1]
                    #                         ]
                    #                         new_input.append(translated_word)
                    #                         new_input.append(translated_word)

                    #                 if output_try != oup:
                    #                     all_correct = False
                    #                     break

                    #                 print("output try: ", output_try)
                    #                 print("correct output: ", oup)

                    #             if all_correct:
                    #                 converged = True
                    #                 break

                    #             # failed
                    #             print("TIMES CHANGED: ", num_times_changed)
                    #             print("Grammar: ", induced_grammar)
                    #             new_examples_str = "\n\n".join(new_examples)
                    #             message.append(
                    #                 {
                    #                     "role": "user",
                    #                     "content": f"That wasn't quite right. Please observe these new examples and try again.\n\nNew Examples:\n"
                    #                     + new_examples_str
                    #                     + f"\nYour previous attempt:\n{induced_grammar}\n\nNew grammar:",
                    #                 }
                    #             )
                    #             completion = get_completion_openai(
                    #                 message, backend, temp=1.0
                    #             )
                    #             induced_grammar = completion["choices"][0]["message"][
                    #                 "content"
                    #             ]
                    #             num_times_changed += 1

                    #     print("TIMES CHANGED: ", num_times_changed)
                    #     print(induced_grammar)
                    #     usage_completion = completion["usage"]["completion_tokens"]
                    #     usage_prompt = completion["usage"]["prompt_tokens"]

                    # else:  # alpaca or llama
                    #     raise NotImplementedError

                    # if return_grammar_only:
                    # return (
                    # induced_grammar,
                    # usage_completion,
                    # usage_prompt,
                    # num_times_changed,
                    # )

                prompt_with_induced_grammar = self.prompt_with_induced_grammar_wrap(
                    induced_grammar, few_shot_examples_str, self.get_input(idx)
                )

                return prompt_with_induced_grammar, usage_completion, usage_prompt
            elif self.prompt_style == "rule_selection":
                rule_selection_prompt = self.get_rule_selection_prompt(idx)
                message = [
                    {"role": "system", "content": PROBLEM_SOLVING_SYSPROMPT},
                    {"role": "user", "content": rule_selection_prompt},
                ]
                completion = get_completion_openai(message, backend, temp=0.0)
                rules = completion["choices"][0]["message"]["content"]
                prompt_with_relevant_rules = self.prompt_with_relevant_rules_wrap(
                    rules, self.get_input(idx)
                )
                return (
                    prompt_with_relevant_rules,
                    completion["usage"]["completion_tokens"],
                    completion["usage"]["prompt_tokens"],
                )

    def induce_word_by_word(
        self,
        idx: int,
        word_lst: List[str],
        backend: str = "gpt-4",
        n_hyps: int = 1,
        rerank_by: str = "p_data_given_hyp_guess",
    ) -> Tuple[str, float, float]:
        rule_set = {}
        usage_completion_all = 0
        usage_prompt_all = 0
        interim_hypotheses = {
            "hypothesis": [],
            "estimated_prob": [],
            "rank": [],
            "task_id": [],
            "for_word": [],
        }
        for word in word_lst:
            few_shot_examples = self.get_examples_with_one_word(word)
            few_shot_examples_str = "\n".join(
                [
                    few_shot_examples_prompt.format(input=i, output=o)
                    for i, o in few_shot_examples
                ]
            )
            word_induction_prompt = prompt_for_grammar_induction_one_word.format(
                word=word, few_shot_examples=few_shot_examples_str
            )

            message = [
                {"role": "system", "content": GRAMMAR_INDUCTION_SYSPROMPT},
                {"role": "user", "content": word_induction_prompt},
            ]
            completion, usage_completion, usage_prompt = self.get_best_grammar(
                word,
                message,
                backend,
                n_hyps,
                rerank_by,
                few_shot_examples,
                return_hyps_ranked=True,
            )
            if isinstance(completion, list):
                alternate_completions = [
                    c[0]["choices"][0]["message"]["content"] for c in completion
                ]
                alternate_probs = [c[1] for c in completion]
                completion = completion[0][0]

                interim_hypotheses["hypothesis"].append(alternate_completions)
                interim_hypotheses["estimated_prob"].append(alternate_probs)
                interim_hypotheses["rank"].append(
                    list(range(len(alternate_completions)))
                )
                interim_hypotheses["task_id"].append([idx] * len(alternate_completions))
                interim_hypotheses["for_word"].append(
                    [word] * len(alternate_completions)
                )

            usage_completion_all += usage_completion
            usage_prompt_all += usage_prompt

            induced_rule = completion["choices"][0]["message"]["content"]
            rule_set[word] = induced_rule

        # save all the interim hypotheses
        for key, value in interim_hypotheses.items():
            # breakpoint()
            interim_hypotheses[key] = [item for sublist in value for item in sublist]
            self.proposed_hypotheses[key].append(interim_hypotheses[key])

        return rule_set, usage_completion_all, usage_prompt_all

    def get_best_grammar(
        self,
        word: str,
        message: str,
        backend: str,
        n_hyps: int,
        rerank_by: Literal[
            "p_data_given_hyp_guess",
            "p_answer_given_hyp_logprobs",
            "p_data_given_hyp_logprobs",
            "ground_truth",
        ],
        few_shot_examples: list,
        return_hyps_ranked: bool = True,
    ):
        usage_completion = 0
        usage_prompt = 0

        few_shot_examples_str = "\n".join(
            [
                few_shot_examples_prompt.format(input=i, output=o)
                for i, o in few_shot_examples
            ]
        )
        assert n_hyps > 0, "n_hyps must be positive"
        if n_hyps == 1:
            # generate one hyp with temp = 0
            completion = get_completion_openai(message, backend, temp=1.0)
            usage_prompt = completion["usage"]["prompt_tokens"]
            usage_completion = completion["usage"]["completion_tokens"]

            return (completion, usage_completion, usage_prompt)
        else:
            # default temp 1.0 for different options
            # use turbo models' backends to get logprobs
            prob_model_names = {
                "gpt-4-0314": "gpt-4-1106-preview",
                "gpt-3.5-turbo-0613": "gpt-3.5-turbo-0613",
            }
            prob_model_name = prob_model_names[backend]
            completions = [
                get_completion_openai(message, prob_model_name, temp=1.0)
                for _ in range(n_hyps)
            ]
            usage_prompt += sum(
                [completion["usage"]["prompt_tokens"] for completion in completions]
            )
            usage_completion += sum(
                [completion["usage"]["completion_tokens"] for completion in completions]
            )

            if rerank_by == "p_data_given_hyp_guess":
                estimate_p_data_prompts = [
                    prompt_for_probability_guess.format(
                        word=word,
                        few_shot_examples=few_shot_examples_str,
                        hypothesis=completion["choices"][0]["message"]["content"],
                    )
                    for completion in completions
                ]
                p_data_given_hyp_guesses = [
                    get_completion_openai(
                        [
                            {
                                "role": "system",
                                "content": "You are a probability estimating system. Your job is to judge how probable data is given an explanation, and answer only with a number from 0 to 1 inclusive.",
                            },
                            {"role": "user", "content": prompt},
                        ],
                        backend,
                        temp=0.0,
                        logprobs=True,
                    )
                    for prompt in estimate_p_data_prompts
                ]

                usage_completion += sum(
                    [
                        completion["usage"]["completion_tokens"]
                        for completion in p_data_given_hyp_guesses
                    ]
                )
                usage_prompt += sum(
                    [
                        completion["usage"]["prompt_tokens"]
                        for completion in p_data_given_hyp_guesses
                    ]
                )

                p_data_given_hyp_guesses = [
                    x["choices"][0]["message"]["content"]
                    for x in p_data_given_hyp_guesses
                ]

                p_data_given_hyp_guesses = [
                    try_convert_to_float(guess) for guess in p_data_given_hyp_guesses
                ]
                # rerank the hypotheses by probability
                completions = [
                    (completion, prob_guess)
                    for prob_guess, completion in sorted(
                        zip(p_data_given_hyp_guesses, completions),
                        key=lambda pair: pair[0],
                        reverse=True,
                    )
                ]

            elif rerank_by == "p_data_given_hyp_logprobs":
                # logprobs - from turbo models
                # oracle condition -> best possible function description in N generated hypotheses
                estimate_p_data_prompts = [
                    prompt_for_probability_logprobs.format(
                        word=word,
                        few_shot_examples=few_shot_examples_str,
                        hypothesis=completion["choices"][0]["message"]["content"],
                    )
                    for completion in completions
                ]
                usage_completion += sum(
                    [
                        completion["usage"]["completion_tokens"]
                        for completion in completions
                    ]
                )
                usage_prompt += sum(
                    [completion["usage"]["prompt_tokens"] for completion in completions]
                )

                logprobs_estimations = [
                    get_completion_openai(
                        prompt, "davinci-002", temp=0.0, logprobs=True, max_tokens=0
                    )
                    for prompt in estimate_p_data_prompts
                ]
                # Exclude the hypothesis itself from the logprobs
                token_offsets = [
                    get_example_start_token(completion, start_indicator="Examples:")
                    for completion in logprobs_estimations
                ]
                sum_logprobs = []
                for i, (offset, completion) in enumerate(
                    zip(token_offsets, logprobs_estimations)
                ):
                    token_logprobs = completion["choices"][0]["logprobs"][
                        "token_logprobs"
                    ]

                    # Error checking
                    if offset >= len(token_logprobs):
                        print(
                            f"  Offset {offset} is out of range for token_logprobs. Skipping this completion."
                        )
                        sum_logprobs.append(-float("inf"))
                        continue

                    # Calculating sum of logprobs
                    sum_logprob = sum(token_logprobs[offset:]) / len(
                        token_logprobs[offset:]
                    )
                    sum_logprobs.append(sum_logprob)

                completions = [
                    (completion, sum_logprob)
                    for sum_logprob, completion in sorted(
                        zip(sum_logprobs, completions),
                        key=lambda pair: pair[0],
                        reverse=True,
                    )
                ]
            elif rerank_by == "p_answer_given_hyp_logprobs":
                estimate_p_data_prompts = [
                    prompt_for_probability_logprobs.format(
                        word=word,
                        few_shot_examples=few_shot_examples_str,
                        hypothesis=completion["choices"][0]["message"]["content"],
                    )
                    for completion in completions
                ]
                usage_completion += sum(
                    [
                        completion["usage"]["completion_tokens"]
                        for completion in completions
                    ]
                )
                usage_prompt += sum(
                    [completion["usage"]["prompt_tokens"] for completion in completions]
                )
                logprobs_estimations = [
                    get_completion_openai(
                        prompt, "davinci-002", temp=0.0, logprobs=True, max_tokens=0
                    )
                    for prompt in estimate_p_data_prompts
                ]
                # Exclude the hypothesis itself from the logprobs
                token_offsets = [
                    get_example_start_token(completion, start_indicator="Examples:")
                    for completion in logprobs_estimations
                ]
                tokens_of_answers_only = []
                for i, (offset, completion) in enumerate(
                    zip(token_offsets, logprobs_estimations)
                ):
                    token_logprobs = completion["choices"][0]["logprobs"][
                        "token_logprobs"
                    ]
                    orig_text = completion["choices"][0]["text"]
                    char_offset = orig_text.find("Examples:") + len("Examples:")
                    orig_text = orig_text[char_offset:]

                    pattern_answer = "Output:\s*([\w\s]+)"
                    answer_tok_ranges = get_start_and_end_tokens(
                        completion, pattern_answer
                    )

                    completion_logprobs_total = []
                    for start_token_ind, end_token_ind in answer_tok_ranges:
                        completion_logprobs_total.append(
                            sum(token_logprobs[start_token_ind:end_token_ind])
                        )

                    tokens_of_answers_only.append(completion_logprobs_total)

                sum_logprobs = [
                    sum(token_logprobs) / len(token_logprobs)
                    for token_logprobs in tokens_of_answers_only
                ]

                completions = [
                    (completion, sum_logprob)
                    for sum_logprob, completion in sorted(
                        zip(sum_logprobs, completions),
                        key=lambda pair: pair[0],
                        reverse=True,
                    )
                ]
            elif rerank_by == "ground_truth":
                parsed_equations = [
                    parse_production_rule(
                        completion["choices"][0]["message"]["content"]
                    )
                    for completion in completions
                ]

                # TODO: this will NOT work with more complex production rules
                # with repeats, contextual rules etc. Need to prompt the model more carefully
                # and also need to parse the production rules more carefully
                # skip the ground truth reranking for colours for now

                icl_example_answers_and_chrfs = [
                    (i, get_ground_truth_answers_vocab(parsed_eq, few_shot_examples))
                    for i, parsed_eq in enumerate(parsed_equations)
                ]

                completions = [
                    (completion, chrf)
                    for completion, (i, (_, chrf)) in sorted(
                        zip(completions, icl_example_answers_and_chrfs),
                        key=lambda pair: pair[1][1][1],
                        reverse=True,
                    )
                ]

            if return_hyps_ranked:
                return completions, usage_completion, usage_prompt

            return completions[0][0], usage_completion, usage_prompt

    def get_zs_cot_prompt(self, idx: int) -> str:
        few_shot_examples = self.get_few_shot_examples(
            idx, return_as_lst=True, no_parse=True
        )
        few_shot_examples_str = "\n".join(
            [
                few_shot_examples_prompt.format(input=i, output=o)
                for i, o in few_shot_examples
            ]
        )
        return cot_zero_shot_prompt["user"].format(
            input=self.get_input(idx),
            few_shot_examples=few_shot_examples_str,
        )

    def get_full_grammar_prompt(self, idx: int) -> str:
        input = self.get_input(idx)
        few_shot_examples = self.get_few_shot_examples(idx)
        if self.split == "miniscan":
            return self.full_grammar_prompt_wrap_miniscan(few_shot_examples, input)

        return self.full_grammar_prompt_wrap(few_shot_examples, input)

    def get_grammar_induction_prompt(self, idx: int, no_parse: bool = False) -> str:
        few_shot_examples = self.get_few_shot_examples(idx, no_parse=no_parse)
        return self.grammar_induction_prompt_wrap(
            few_shot_examples,
        )

    def get_rule_selection_prompt(self, idx: int) -> str:
        input = self.get_input(idx)
        return self.rule_induction_prompt_wrap(input)

    def get_system_prompt(self) -> str:
        if self.prompt_style == "full_grammar":
            return prompt_with_true_grammar["system"]
        elif self.prompt_style == "grammar_induction":
            return prompt_for_grammar_induction["system"]

        return base_prompt["system"]

    def get_repeat_prompt(
        self,
        induced_grammar: str,
        schedule: str = "random",
        num_examples_to_get: Optional[int] = None,
        return_examples_only: bool = False,
    ) -> str:
        if schedule == "random":
            rand_seed = random.randint(0, 1000)
            num_examples = (
                num_examples_to_get
                if num_examples_to_get is not None
                else self.num_few_shot_examples
            )
            new_few_shot_examples = self.get_few_shot_examples(
                idx=rand_seed, use_few_shot_minset=False, num_examples=num_examples
            )
            if return_examples_only:
                return new_few_shot_examples
        elif schedule == "target_hard_words":
            hard_words = ["bluf", "walm"]
            rand_seed = random.randint(0, 1000)
            num_examples = (
                num_examples_to_get
                if num_examples_to_get is not None
                else self.num_few_shot_examples
            )
            new_few_shot_examples = self.get_few_shot_examples(
                idx=rand_seed, use_few_shot_minset=False, num_examples=num_examples
            ).split("\n\n")
            # filter out examples that don't contain hard words
            new_few_shot_examples = [
                example
                for example in new_few_shot_examples
                if any([word in example for word in hard_words])
            ]
            num_usable = len(new_few_shot_examples)
            i = 0
            while num_usable < num_examples:
                i += 1
                new_few_shot_examples_i = self.get_few_shot_examples(
                    idx=rand_seed + i,
                    use_few_shot_minset=False,
                    num_examples=num_examples,
                ).split("\n\n")
                new_few_shot_examples_i = [
                    example
                    for example in new_few_shot_examples_i
                    if any([word in example for word in hard_words])
                ]
                num_usable += len(new_few_shot_examples_i)
                new_few_shot_examples.extend(new_few_shot_examples_i)

            if return_examples_only:
                return new_few_shot_examples

            new_few_shot_examples = "\n\n".join(new_few_shot_examples)

        else:
            raise NotImplementedError
        return prompt_for_grammar_induction["user_repeat"].format(
            induced_grammar=induced_grammar, few_shot_examples=new_few_shot_examples
        )

    @staticmethod
    def standard_prompt_wrap(few_shot_examples: str, input: str) -> str:
        return base_prompt["user"].format(
            few_shot_examples=few_shot_examples, input=input
        )

    @staticmethod
    def full_grammar_prompt_wrap(few_shot_examples: str, input: str) -> str:
        return prompt_with_true_grammar["user"].format(
            few_shot_examples=few_shot_examples, input=input
        )

    @staticmethod
    def full_grammar_prompt_wrap_miniscan(few_shot_examples: str, input: str) -> str:
        return prompt_with_true_grammar_miniscan["user"].format(
            few_shot_examples=few_shot_examples, input=input
        )

    @staticmethod
    def grammar_induction_prompt_wrap(few_shot_examples: str) -> str:
        return prompt_for_grammar_induction["user_new"].format(
            few_shot_examples=few_shot_examples
        )

    @staticmethod
    def prompt_with_induced_grammar_wrap(
        induced_grammar: str, few_shot_examples: str, input: str
    ) -> str:
        return prompt_for_grammar_induction["user_followup"].format(
            induced_grammar=induced_grammar,
            input=input,
            few_shot_examples=few_shot_examples,
        )

    @staticmethod
    def rule_induction_prompt_wrap(input: str) -> str:
        return prompt_for_rule_selection["user"].format(input=input)

    @staticmethod
    def prompt_with_relevant_rules_wrap(rules: str, input: str) -> str:
        return prompt_for_rule_selection["user_followup"].format(
            rules=rules, input=input
        )

    @staticmethod
    def few_shot_examples_wrap(few_shot_inputs: List, few_shot_outputs: List) -> str:
        examples = ""
        for inp, oup in zip(few_shot_inputs, few_shot_outputs):
            example = few_shot_examples_prompt.format(input=inp, output=oup)
            examples += example + "\n"
        return examples
