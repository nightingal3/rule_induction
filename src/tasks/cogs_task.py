from typing import List, Literal

import pandas as pd
import random

from src.prompts.cogs_prompts import * 
from src.task import BaseTask
from src.prompt_openai import get_completion

class CogsTask(BaseTask):
    def __init__(self, train_file: str, test_file: str, prompt_style: Literal["base", "full_grammar", "grammar_induction"], split: Literal["cp_recursion", "prim_to_subj_common", "exposure_example_obj_proper", "obj_to_subj_common", "only_seen_as_unacc_subj_as_obj_omitted_transitive_subj", "simple"], num_few_shot_examples: int = 5, nonce: bool = False, **kwargs) -> None:
        self.train_file = train_file
        self.test_file = test_file
        self.train_data = pd.read_csv(train_file, sep="\t")
        self.test_data = pd.read_csv(test_file, sep="\t")
        if split != "simple":
            self.test_data = self.test_data.loc[self.test_data["split"] == split]
        self.prompt_style = prompt_style
        self.num_few_shot_examples = num_few_shot_examples
        self.is_nonce = nonce

    def __len__(self) -> int:
        return len(self.test_data)

    def get_input(self, idx: int) -> str:
        return self.test_data.iloc[idx]["sentence"]
    
    def get_few_shot_examples(self, idx: int) -> str:
        # get some random examples based on the idx (repeatable)
        random.seed(idx)
        indices = random.sample(range(len(self.train_data)), self.num_few_shot_examples)
        few_shot_inputs = [self.train_data.iloc[i]["sentence"] for i in indices]
        few_shot_outputs = [self.train_data.iloc[i]["parse"] for i in indices]
        few_shot_formatted = self.few_shot_examples_wrap(few_shot_inputs, few_shot_outputs)
        return few_shot_formatted
    
    def validate(self, idx: int, output: str) -> bool:
        output_text = output["choices"][0]["message"]["content"]
        output_actions = output_text.split("Output: ")[-1].strip()
        return output_actions == self.test_data.iloc[idx]["parse"]
    
    def get_standard_prompt(self, idx: int) -> str:
        input = self.get_input(idx)
        few_shot_examples = self.get_few_shot_examples(idx)
        return self.standard_prompt_wrap(few_shot_examples, input)
    
    def get_special_prompt(self, idx: int, backend: str = "gpt-4") -> str:
        if self.prompt_style == "full_grammar":
            return self.get_full_grammar_prompt(idx)
        else:
            grammar_induction_prompt = self.get_grammar_induction_prompt(idx)
            message = [
                {"role": "system", "content": GRAMMAR_INDUCTION_SYSPROMPT},
                {"role": "user", "content": grammar_induction_prompt},
            ]
            few_shot_examples = self.get_few_shot_examples(idx + 1) # selecting some different examples for second step
            induced_grammar = get_completion(message, backend, temp=0.0)["choices"][0]["message"]["content"]
            prompt_with_induced_grammar = self.prompt_with_induced_grammar_wrap(induced_grammar, few_shot_examples, self.get_input(idx))
            return prompt_with_induced_grammar
        
    def get_full_grammar_prompt(self, idx: int) -> str:
        input = self.get_input(idx)
        few_shot_examples = self.get_few_shot_examples(idx)
        return self.full_grammar_prompt_wrap(few_shot_examples, input)
    
    def get_grammar_induction_prompt(self, idx: int) -> str:
        few_shot_examples = self.get_few_shot_examples(idx)
        return self.grammar_induction_prompt_wrap(few_shot_examples,)
    
        
    def get_system_prompt(self) -> str:
        if self.prompt_style == "full_grammar":
            return prompt_with_true_grammar["system"]
        elif self.prompt_style == "grammar_induction":
            return prompt_for_grammar_induction["system"]
            
        return base_prompt["system"]
    
    @staticmethod
    def standard_prompt_wrap(few_shot_examples: str, input: str) -> str:
        return base_prompt["user"].format(few_shot_examples=few_shot_examples, input=input)
    
    @staticmethod
    def full_grammar_prompt_wrap(few_shot_examples: str, input: str) -> str:
        return prompt_with_true_grammar["user"].format(few_shot_examples=few_shot_examples, input=input)
    
    @staticmethod
    def grammar_induction_prompt_wrap(few_shot_examples: str) -> str:
        return prompt_for_grammar_induction["user"].format(few_shot_examples=few_shot_examples)
    
    @staticmethod
    def prompt_with_induced_grammar_wrap(induced_grammar: str, few_shot_examples: str, input: str) -> str:
        return prompt_for_grammar_induction["user_followup"].format(induced_grammar=induced_grammar, input=input, few_shot_examples=few_shot_examples)
    
    @staticmethod
    def few_shot_examples_wrap(few_shot_inputs: List, few_shot_outputs: List) -> str:
        examples = ""
        for inp, oup in zip(few_shot_inputs, few_shot_outputs):
            example = few_shot_examples_prompt.format(input=inp, output=oup)
            examples += example
        return examples