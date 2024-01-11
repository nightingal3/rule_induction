from typing import List, Literal, Tuple, Optional

import pandas as pd
import random
import json
import jsonlines
import numpy as np
import os
import ast

from src.prompts.arc_prompts import * 
from src.task import BaseTask
from src.prompt_openai import get_completion_openai

class ArcTask(BaseTask):

    task_type = "arc"
    
    def __init__(self, split: str, prompt_style: Literal["base", "full_grammar", "grammar_induction"], num_few_shot_examples: int = 5, few_shot_min_set: bool = False, grammar_induction_loop: bool = False, **kwargs) -> None:
        self.split = split
        examples_path = f"./data/ConceptARC/{split}"
        self.examples = {}
        self.test_examples = []
        self.test_examples_ids = []
        for file in os.listdir(examples_path):
            if file.endswith(".json"):
                example_id = file.split(".json")[0].split(self.split)[1]
                example = json.load(open(os.path.join(examples_path, file)))
                for test_example in example["test"]:
                    self.test_examples.append(test_example)
                    self.test_examples_ids.append(example_id)
                self.examples[example_id] = example

        self.prompt_style = prompt_style
        self.num_few_shot_examples = num_few_shot_examples
        # if prompt style is base, this will just select the min set examples.
        # if prompt style is full_grammar, this will use the example parses as well.
        self.use_few_shot_minset = few_shot_min_set
        if self.use_few_shot_minset:
            self.min_examples = pd.read_csv("./data/colours/all_commands_minset.csv")

        self.cached_induced_grammars = []
        if os.path.exists("./data/colours/gpt_4_induced_grammars.jsonl"):
            with jsonlines.open("./data/colours/gpt_4_induced_grammars.jsonl", "r") as f:
                for line in f:
                    self.cached_induced_grammars.append(line)

        self.grammar_induction_loop = grammar_induction_loop

        self.rules = {
            "AboveBelow": {
                "Minimal": "You will see a horizontal line of one value dividing the input grid. Copy only the value of the shape above that line. You should make the output dimensions the same as the shape above the line. For instance, if the shape is a 2x2 square of '4's, your output should be a 2x2 square of '4's.",
                "1": "You will see a dashed or dotted horizontal line. You should copy everything above the line and the line itself, but fill in with 0s everything below the line.",
                "2": "You will see a square of 3s. Fill in with 0s everything that’s above the square of 3s.",
                "3": "You will see a shape with a line hovering above it. Move the line so that it touches the top of the shape, and there is no empty space between the line and the shape.",
                "4": "You will see a horizontal line dividing the grid. For the shapes above the line, fill in the leftmost value of the shape with the same numerical value as the value in the line.",
                "5": "In each column that is not filled with 0s, there is a special value that appears only once. You should fill in every cell above this special value with the same thing as the special value, and keep everything else the same. If the column is already filled with all the same value, keep it the same.",
                "6": "In each column that is not filled with 0s, there is a special value that appears only once. You should fill in every cell below this special value with the same thing as the special value, and keep everything else the same. If the column is already filled with all the same value, keep it the same.",
                "7": "You will see one or more shapes with a top half and a bottom half marked by diofferent values. You should move the top portion below the bottom portion so that they just touch. Do not rotate or flip the bottom portion.",
                "8": "You will see a squiggly line of 8s within a grid of many different values, arranged horizontally. In the output, replace all portions of the grid with 0s except for the portion of the squiggly line of 8s that’s below the portion of the grid that has the value 3. Also erase the 3s and only keep the squiggly line.",
                "9": "You will see a horizontal line of one value dividing the input grid. Fill in with 0s everything below the line. Do not erase the line itself.",
                "10": "You will see a diagonal line consisting of one value. Fill in everything below that line with the same value so that the final result looks like a staircase."
            }
        }

    def __len__(self) -> int:
        return len(self.test_examples)

    # this dataset has groups of 3 inputs 
    def get_few_shot_examples(self, idx: int, add_example_input: bool = False, **kwargs) -> str:
        example_id = self.test_examples_ids[idx]
        demonstrations = self.examples[example_id]["train"]
        inputs = [x["input"] for x in demonstrations]
        outputs = [x["output"] for x in demonstrations]

        prompt = self.few_shot_examples_wrap(inputs, outputs)
        if add_example_input:
            example_test_input = self.test_examples[idx]["input"]
            prompt = f"{prompt}\n\nHere's an example of an input this rule would be applied to. Please make sure that your rule also generalizes to this input:\n\n{example_test_input}"

        return prompt
    
    def get_input(self, idx: int) -> str: # TODO: minimal example not included
        return self.test_examples[idx]["input"]
    
    def get_answer(self, idx: int) -> str:
        return self.test_examples[idx]["output"]
    
    def validate(self, idx: int, output_actions: str) -> bool:
        if "Output:" in output_actions:
            output_actions = output_actions.split("Output:")[1].strip()
        try:
            print(output_actions)
            output_lst = ast.literal_eval(output_actions)
        except:
            try:
                output_actions = output_actions.split("Input:")[0].strip()
                output_lst = ast.literal_eval(output_actions)
            except:
                print(f"Error parsing output actions on {idx}. Returning False for now, validate manually.")
                return False
        return output_lst == self.get_answer(idx)
    
    def get_standard_prompt(self, idx: int) -> Tuple[str, str]:
        input = self.get_input(idx)
        few_shot_examples = self.get_few_shot_examples(idx)
        return self.standard_prompt_wrap(few_shot_examples, input), self.test_examples_ids[idx]
    
    def get_special_prompt(self, idx: int, backend: str = "gpt-4", return_grammar_only: bool = False, use_cached: bool = False) -> Tuple[str, int, int]:
        if self.prompt_style == "full_grammar":
            return self.get_full_grammar_prompt(idx), 0, 0, self.test_examples_ids[idx]
        else:
            if self.prompt_style == "grammar_induction":
                few_shot_examples = self.get_few_shot_examples(idx, add_example_input=True)
                if use_cached:
                    induced_grammar = self.cached_induced_grammars[0]["grammar"]
                    usage_completion = 0
                    usage_prompt = 0
                else:
                    grammar_induction_prompt = self.get_grammar_induction_prompt(idx, no_parse=True)
                    if "gpt" in backend:
                        message = [
                            {"role": "system", "content": GRAMMAR_INDUCTION_SYSPROMPT},
                            {"role": "user", "content": grammar_induction_prompt},
                        ]
                        completion = get_completion_openai(message, backend, temp=0.0)
                        induced_grammar = completion["choices"][0]["message"]["content"]

                        converged = False
                        num_times_changed = 0

                        while self.grammar_induction_loop is True and not converged:
                            repeat_prompt = self.get_repeat_prompt(induced_grammar, schedule="target_hard_words", num_examples_to_get=3)
                            message = [
                                {"role": "system", "content": GRAMMAR_INDUCTION_SYSPROMPT},
                                {"role": "user", "content": repeat_prompt}
                            ]
                            completion = get_completion_openai(message, backend, temp=1)
                            induced_grammar_new = completion["choices"][0]["message"]["content"]
                            if "no changes" in induced_grammar_new.lower():
                                converged = True
                            else:
                                induced_grammar = induced_grammar_new
                                num_times_changed += 1
                        
                        print("TIMES CHANGED: ", num_times_changed)
                        print(induced_grammar)
                        usage_completion = completion["usage"]["completion_tokens"]
                        usage_prompt = completion["usage"]["prompt_tokens"]

                    else: # alpaca or llama
                        raise NotImplementedError
                    if return_grammar_only:
                        return induced_grammar, usage_completion, usage_prompt, self.test_examples_ids[idx]
                    
                prompt_with_induced_grammar = self.prompt_with_induced_grammar_wrap(induced_grammar, few_shot_examples, self.get_input(idx))
                return prompt_with_induced_grammar, usage_completion, usage_prompt, self.test_examples_ids[idx]
        
            elif self.prompt_style == "rule_selection":
                rule_selection_prompt = self.get_rule_selection_prompt(idx)
                message = [
                        {"role": "system", "content": PROBLEM_SOLVING_SYSPROMPT},
                        {"role": "user", "content": rule_selection_prompt},
                    ]
                completion = get_completion_openai(message, backend, temp=0.0)
                rules = completion["choices"][0]["message"]["content"]
                prompt_with_relevant_rules = self.prompt_with_relevant_rules_wrap(rules, self.get_input(idx))
                return prompt_with_relevant_rules, completion["usage"]["completion_tokens"], completion["usage"]["prompt_tokens"]
        
    def get_full_grammar_prompt(self, idx: int) -> str:
        input = self.get_input(idx)
        few_shot_examples = self.get_few_shot_examples(idx)
        return self.full_grammar_prompt_wrap(idx, few_shot_examples, input)
    
    def get_grammar_induction_prompt(self, idx: int, no_parse: bool = False) -> str:
        few_shot_examples = self.get_few_shot_examples(idx, no_parse=no_parse)
        return self.grammar_induction_prompt_wrap(idx, few_shot_examples)
    
    def get_rule_selection_prompt(self, idx: int) -> str:
        input = self.get_input(idx)
        return self.rule_induction_prompt_wrap(input)
    
    def get_system_prompt(self) -> str:
        if self.prompt_style == "full_grammar":
            return prompt_with_true_grammar["system"]
        elif self.prompt_style == "grammar_induction":
            return prompt_for_grammar_induction["system"]
            
        return base_prompt["system"]
    
    def get_repeat_prompt(self, induced_grammar: str, schedule: str = "random", num_examples_to_get: Optional[int] = None) -> str:
        if schedule == "random":
            rand_seed = random.randint(0, 1000)
            num_examples = num_examples_to_get if num_examples_to_get is not None else self.num_few_shot_examples
            new_few_shot_examples = self.get_few_shot_examples(idx=rand_seed, use_few_shot_minset=False, num_examples=num_examples)
        elif schedule == "target_hard_words":
            hard_words = ["bluf", "walm"]
            rand_seed = random.randint(0, 1000)
            num_examples = num_examples_to_get if num_examples_to_get is not None else self.num_few_shot_examples
            new_few_shot_examples = self.get_few_shot_examples(idx=rand_seed, use_few_shot_minset=False, num_examples=num_examples).split("\n\n")
            # filter out examples that don't contain hard words
            new_few_shot_examples = [example for example in new_few_shot_examples if any([word in example for word in hard_words])]
            num_usable = len(new_few_shot_examples)
            i = 0
            while num_usable < num_examples:
                i += 1
                new_few_shot_examples_i = self.get_few_shot_examples(idx=rand_seed + i, use_few_shot_minset=False, num_examples=num_examples).split("\n\n")
                new_few_shot_examples_i = [example for example in new_few_shot_examples_i if any([word in example for word in hard_words])]
                num_usable += len(new_few_shot_examples_i)
                new_few_shot_examples.extend(new_few_shot_examples_i)
            new_few_shot_examples = "\n\n".join(new_few_shot_examples)
        else:
            raise NotImplementedError
        return prompt_for_grammar_induction["user_repeat"].format(induced_grammar=induced_grammar, few_shot_examples=new_few_shot_examples)

        
    def full_grammar_prompt_wrap(self, idx: int, few_shot_examples: str, input: str) -> str:
        rule = self.rules[self.split][self.test_examples_ids[idx]]
        return prompt_with_true_grammar["user"].format(induced_grammar=rule, few_shot_examples=few_shot_examples, input=input)
    

    def grammar_induction_prompt_wrap(self, idx: int, few_shot_examples: str) -> str:
        # get one test input only so that the rule is more general and not specific to demonstrations
        example_input = self.get_input(idx)
        return prompt_for_grammar_induction["user"].format(few_shot_examples=few_shot_examples, example_input=example_input)
    
    @staticmethod
    def standard_prompt_wrap(few_shot_examples: str, input: str) -> str:
        return base_prompt["user"].format(few_shot_examples=few_shot_examples, input=input)

    @staticmethod
    def prompt_with_induced_grammar_wrap(induced_grammar: str, few_shot_examples: str, input: str) -> str:
        return prompt_for_grammar_induction["user_followup"].format(induced_grammar=induced_grammar, input=input, few_shot_examples=few_shot_examples)

    @staticmethod
    def rule_induction_prompt_wrap(input: str) -> str:
        return prompt_for_rule_selection["user"].format(input=input)
    
    @staticmethod 
    def prompt_with_relevant_rules_wrap(rules: str, input: str) -> str:
        return prompt_for_rule_selection["user_followup"].format(rules=rules, input=input)
    
    @staticmethod
    def few_shot_examples_wrap(few_shot_inputs: List, few_shot_outputs: List) -> str:
        examples = ""
        for inp, oup in zip(few_shot_inputs, few_shot_outputs):
            example = few_shot_examples_prompt.format(input=inp, output=oup)
            examples += example + "\n"
        return examples