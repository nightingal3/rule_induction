import pandas as pd
import subprocess
import re
import ast
import numpy as np
from typing import List


def find_examples_with_pattern(data: pd.DataFrame, pattern: str) -> pd.DataFrame:
    return data[data["patterns"].apply(lambda x: pattern in x)]


def find_examples_with_all_patterns(data, patterns: List[str]) -> pd.DataFrame:
    return data[data["patterns"].apply(lambda x: set(patterns).issubset(set(x)))]


def find_matching_dict_entries(input_str: str, dictionary: List[str]) -> dict:
    input_str_words = input_str.split()
    return {k: v for k, v in dictionary.items() if k in input_str_words}


def try_convert_to_float(val: str):
    try:
        return float(val)
    except:
        return 0


def edit_distance(word1: str, word2: str) -> int:
    min_distance_table = [
        [0 for _ in range(len(word2) + 1)] for _ in range(len(word1) + 1)
    ]
    for i in range(len(word1) + 1):
        min_distance_table[i][0] = i
    for j in range(len(word2) + 1):
        min_distance_table[0][j] = j

    for i in range(1, len(word1) + 1):
        for j in range(1, len(word2) + 1):
            if word1[i - 1] == word2[j - 1]:
                min_distance_table[i][j] = min_distance_table[i - 1][j - 1]
            else:
                min_distance_table[i][j] = 1 + min(
                    min_distance_table[i - 1][j],  # delete
                    min_distance_table[i][j - 1],  # insert
                    min_distance_table[i - 1][j - 1],  # substitute
                )

    return min_distance_table[-1][-1]


def get_job_info_by_name(job_name: str) -> str:
    cmd = "squeue --format=%j,%i,%N"
    proc = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    output, err = proc.communicate()

    if err:
        raise Exception(err)

    for line in output.decode("utf-8").split("\n"):
        if line.startswith(job_name + ","):
            _, job_id, node = line.split(",")
            return job_id, node

    print(f"Job {job_name} not found")
    return None
