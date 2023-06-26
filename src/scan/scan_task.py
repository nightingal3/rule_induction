from typing import List, Literal

import pandas as pd

from prompts import * 
from ..task import BaseTask

class ScanTask(BaseTask):
    def __init__(self, train_file: str, test_file: str, prompt_style: Literal["base", "full_grammar", "grammar_induction"] ) -> None:
        self.train_file = train_file
        self.test_file = test_file
        self.train_data = pd.read_csv(train_file)
        self.test_data = pd.read_csv(test_file)
        self.prompt_style = prompt_style
        
    def __len__(self) -> int:
        return len(self.test_data)

    def get_input(self, idx: int) -> str:
        return self.test_data.iloc[idx]["commands"]
    
    def validate(self, idx: int, output: str) -> bool:
        output_actions = output.split("Output: ")[-1].strip()
        return output_actions == self.test_data.iloc[idx]["actions"]
    
    @staticmethod
    def standard_prompt_wrap(few_shot_examples: str, input: str) -> str:
        return base_prompt["user"].format(few_shot_examples=few_shot_examples, input=input)
    