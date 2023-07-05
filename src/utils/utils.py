import pandas as pd
from typing import List

def find_examples_with_pattern(data: pd.DataFrame, pattern: str) -> pd.DataFrame:
    return data[data["patterns"].apply(lambda x: pattern in x)]

def find_examples_with_all_patterns(data, patterns: List[str]) -> pd.DataFrame:
    return data[data["patterns"].apply(lambda x: set(patterns).issubset(set(x)))]