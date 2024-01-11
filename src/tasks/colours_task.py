from typing import List, Literal, Tuple, Optional

import pandas as pd
import random
import json
import jsonlines
import numpy as np
import os
from collections import defaultdict

from src.prompts.colours_prompts import *
from src.task import BaseTask
from src.prompt_openai import get_completion_openai


class ColoursTask(BaseTask):
    task_type = "io"

    def __init__(
        self,
        rules_file: str,
        prompt_style: Literal["base", "full_grammar", "grammar_induction"],
        num_few_shot_examples: int = 5,
        nonce: bool = False,
        few_shot_min_set: bool = False,
        grammar_induction_loop: bool = False,
        **kwargs,
    ) -> None:
        self.rules = pd.read_csv(rules_file)
        self.train_data, self.test_data = self.generate_examples()

        self.prompt_style = prompt_style
        self.num_few_shot_examples = num_few_shot_examples
        self.is_nonce = nonce
        # if prompt style is base, this will just select the min set examples.
        # if prompt style is full_grammar, this will use the example parses as well.
        self.use_few_shot_minset = few_shot_min_set
        if self.use_few_shot_minset:
            self.min_examples = pd.read_csv("./data/colours/all_commands_minset.csv")

        self.cached_induced_grammars = []
        if os.path.exists("./data/colours/gpt_4_induced_grammars.jsonl"):
            with jsonlines.open(
                "./data/colours/gpt_4_induced_grammars.jsonl", "r"
            ) as f:
                for line in f:
                    self.cached_induced_grammars.append(line)

        self.grammar_induction_loop = grammar_induction_loop

    def generate_examples(self, data_size=1000) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if os.path.exists("./data/colours/train_data.csv") and os.path.exists(
            "./data/colours/test_data.csv"
        ):
            train_data = pd.read_csv("./data/colours/train_data.csv")
            test_data = pd.read_csv("./data/colours/test_data.csv")
        else:
            colour_mappings = self.rules.loc[self.rules["rule_type"] == "colour"]
            number_mappings = self.rules.loc[self.rules["rule_type"] == "number"]
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
        no_parse: bool = False,
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
                few_shot_formatted = self.few_shot_examples_wrap(
                    few_shot_inputs, few_shot_outputs
                )
            else:
                all_examples = self.min_examples["example_parse_few_shot"]
                if return_as_lst:
                    return all_examples
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
                return few_shot_inputs, few_shot_outputs
            few_shot_formatted = self.few_shot_examples_wrap(
                few_shot_inputs, few_shot_outputs
            )

        return few_shot_formatted

    def validate(self, idx: int, output_text: str) -> bool:
        output_actions = output_text.split("Output: ")[-1].strip()
        return output_actions == self.test_data.iloc[idx]["actions"]

    def get_standard_prompt(self, idx: int) -> str:
        input = self.get_input(idx)
        few_shot_examples = self.get_few_shot_examples(idx)
        return self.standard_prompt_wrap(few_shot_examples, input), str(idx)

    def get_special_prompt(
        self,
        idx: int,
        backend: str = "gpt-4",
        return_grammar_only: bool = False,
        use_cached: bool = False,
        rejection_sampling: bool = False,
        **kwargs,
    ) -> Tuple[str, int, int]:
        if self.prompt_style == "full_grammar":
            return self.get_full_grammar_prompt(idx), 0, 0
        else:
            if self.prompt_style == "grammar_induction":
                few_shot_examples = self.get_few_shot_examples(
                    idx + 1
                )  # selecting some different examples for second step
                if use_cached:
                    induced_grammar = self.cached_induced_grammars[0]["grammar"]
                    usage_completion = 0
                    usage_prompt = 0
                else:
                    grammar_induction_prompt = self.get_grammar_induction_prompt(
                        idx, no_parse=True
                    )
                    if "gpt" in backend:
                        message = [
                            {"role": "system", "content": GRAMMAR_INDUCTION_SYSPROMPT},
                            {"role": "user", "content": grammar_induction_prompt},
                        ]
                        if not rejection_sampling:
                            completion = get_completion_openai(
                                message, backend, temp=0.0
                            )
                        else:
                            completion = get_completion_openai(
                                message, backend, temp=1.0
                            )

                        induced_grammar = completion["choices"][0]["message"]["content"]

                        converged = False
                        num_times_changed = 0
                        while (self.grammar_induction_loop and not converged) or (
                            rejection_sampling and not converged
                        ):
                            repeat_prompt = self.get_repeat_prompt(
                                induced_grammar,
                                schedule="target_hard_words",
                                num_examples_to_get=5,
                            )
                            if self.grammar_induction_loop:
                                if num_times_changed == 0:
                                    message = [
                                        {
                                            "role": "system",
                                            "content": GRAMMAR_INDUCTION_SYSPROMPT,
                                        },
                                        {"role": "user", "content": repeat_prompt},
                                    ]
                                else:
                                    new_examples = self.get_repeat_prompt(
                                        induced_grammar,
                                        schedule="target_hard_words",
                                        num_examples_to_get=5,
                                        return_examples_only=True,
                                    )
                                    new_examples_str = "\n\n".join(new_examples)
                                    # if there are too many previous revisions (> 5, truncate to 5)
                                    if len(message) >= 5:
                                        message = message[-4:]
                                        message.insert(
                                            0,
                                            {
                                                "role": "system",
                                                "content": GRAMMAR_INDUCTION_SYSPROMPT,
                                            },
                                        )
                                    message.append(
                                        {
                                            "role": "user",
                                            "content": f"That wasn't quite right. Please observe these new examples and try again.\n\nNew Examples:\n"
                                            + new_examples_str
                                            + f"\nYour previous attempt:\n{induced_grammar}\n\nNew grammar:",
                                        }
                                    )

                                completion = get_completion_openai(
                                    message, backend, temp=1
                                )
                                induced_grammar_new = completion["choices"][0][
                                    "message"
                                ]["content"]
                                if "no changes" in induced_grammar_new.lower():
                                    converged = True
                                else:
                                    induced_grammar = induced_grammar_new
                                    num_times_changed += 1
                            elif rejection_sampling:
                                new_examples = self.get_repeat_prompt(
                                    induced_grammar,
                                    schedule="target_hard_words",
                                    num_examples_to_get=5,
                                    return_examples_only=True,
                                )
                                rules_split = induced_grammar.split("\n")
                                rules_dict = defaultdict(str)
                                for rule in rules_split:
                                    if "->" not in rule:
                                        continue
                                    try:
                                        lhs, rhs = rule.split("->")[:2]
                                    except:
                                        print("unparseable rule: ", rule)
                                        continue
                                    lhs = lhs.strip()
                                    rhs = rhs.strip()
                                    # remove numbering such as 1. or 2. or 3.
                                    lhs = lhs.split(". ")[-1]
                                    # remove left/right brackets

                                    if lhs not in rules_dict:
                                        rules_dict[lhs] = rhs

                                # use string substitution to test the rules. If the rules are incorrect, generate new rules.
                                all_correct = False
                                for example in new_examples:
                                    inp, oup = example.split("\n")
                                    inp = inp.split(": ")[1].strip()
                                    oup = oup.split(": ")[1].strip()

                                    inp = inp.split(". ")[-1]
                                    output_try = inp

                                    # replace all the lhs with rhs
                                    for lhs, rhs in rules_dict.items():
                                        if "bluf" in lhs or "walm" in lhs:
                                            continue  # repeat rules, do them last
                                        output_try = output_try.replace(lhs, rhs)

                                    # repeat rules
                                    bluf_rule = rules_dict["bluf"]
                                    walm_rule = rules_dict["walm"]
                                    keywords_bluf = [
                                        "repeat",
                                        "twice",
                                        "2",
                                        "two",
                                        "double",
                                        "before",
                                    ]
                                    keywords_walm = [
                                        "repeat",
                                        "thrice",
                                        "3",
                                        "three",
                                        "triple",
                                        "before",
                                    ]
                                    # if none of the keywords are present, it's probably wrong
                                    if not any(
                                        [
                                            keyword in bluf_rule
                                            for keyword in keywords_bluf
                                        ]
                                    ) or not any(
                                        [
                                            keyword in walm_rule
                                            for keyword in keywords_walm
                                        ]
                                    ):
                                        all_correct = False
                                        break

                                    # TODO: we can replace them but it's probably correct if it has the keywords
                                    split_output = output_try.split(" ")
                                    output_try_new = []
                                    # repeat the previous thing once more, or twice more
                                    for i, word in enumerate(split_output):
                                        if word != "bluf" and word != "walm":
                                            new_input.append(word)
                                        if word == "bluf":
                                            translated_word = rules_dict[
                                                split_output[i - 1]
                                            ]
                                            new_input.append(translated_word)
                                        elif word == "walm":
                                            translated_word = rules_dict[
                                                split_output[i - 1]
                                            ]
                                            new_input.append(translated_word)
                                            new_input.append(translated_word)

                                    if output_try != oup:
                                        all_correct = False
                                        break

                                    print("output try: ", output_try)
                                    print("correct output: ", oup)

                                if all_correct:
                                    converged = True
                                    break

                                # failed
                                print("TIMES CHANGED: ", num_times_changed)
                                print("Grammar: ", induced_grammar)
                                new_examples_str = "\n\n".join(new_examples)
                                message.append(
                                    {
                                        "role": "user",
                                        "content": f"That wasn't quite right. Please observe these new examples and try again.\n\nNew Examples:\n"
                                        + new_examples_str
                                        + f"\nYour previous attempt:\n{induced_grammar}\n\nNew grammar:",
                                    }
                                )
                                completion = get_completion_openai(
                                    message, backend, temp=1.0
                                )
                                induced_grammar = completion["choices"][0]["message"][
                                    "content"
                                ]
                                num_times_changed += 1

                        print("TIMES CHANGED: ", num_times_changed)
                        print(induced_grammar)
                        usage_completion = completion["usage"]["completion_tokens"]
                        usage_prompt = completion["usage"]["prompt_tokens"]

                    else:  # alpaca or llama
                        raise NotImplementedError
                    if return_grammar_only:
                        return (
                            induced_grammar,
                            usage_completion,
                            usage_prompt,
                            num_times_changed,
                        )

                prompt_with_induced_grammar = self.prompt_with_induced_grammar_wrap(
                    induced_grammar, few_shot_examples, self.get_input(idx)
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

    def get_full_grammar_prompt(self, idx: int) -> str:
        input = self.get_input(idx)
        few_shot_examples = self.get_few_shot_examples(idx)
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
