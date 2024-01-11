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


def parse_polynomial(orig_polynomial: str):
    # This function was written by chatgpt and postedited by me. Use at own risk but seems to be working
    # Remove spaces and 'y=' part if present
    # model outputs this when it declines to answer
    # return {"0": None, "1": None}

    if "y" in orig_polynomial:
        polynomial = orig_polynomial.split("=")[-1].split(".")[0].split("\n")[0]
    else:
        polynomial = orig_polynomial.split(".")[0].split("\n")[0]
    polynomial = polynomial.replace("-", "+-")
    polynomial = polynomial.replace(" ", "")

    # Split the polynomial into terms
    terms = polynomial.split("+")
    terms = [term.strip() for term in terms if term != ""]

    # Regular expression to match coefficients and powers
    regex = r"(-?\d*)(x\^?(\d*))?"

    # Dictionary to store coefficients with their degrees
    coefficients = {}

    for term in terms:
        match = re.match(regex, term)
        if match:
            # Extract coefficient and power
            coeff_str, _, power_str = match.groups()

            # Set default values if missing
            try:
                coeff = int(coeff_str) if coeff_str else 1
                power = int(power_str) if power_str else 1 if "x" in term else 0
            except:
                print("=== Error parsing: ", orig_polynomial)
                # if we can't parse the polynomial, return None
                return {"0": None, "1": None}

            if abs(coeff) > 100:
                print("=== Outlier example: ", orig_polynomial)
                print("=== Coefficient: ", coeff)

            coefficients[power] = coeff

    return coefficients


def equation_accuracy(
    df: pd.DataFrame, eq_hyp_col: str, eq_col: str, degree: int
) -> dict:
    df["lst_eq_hyp"] = df[eq_hyp_col].apply(
        lambda x: [x[i] if i in x.keys() else None for i in range(degree + 1)][::-1]
    )
    df[eq_col] = df[eq_col].apply(ast.literal_eval)
    is_equal = df.apply(lambda x: x["lst_eq_hyp"] == x[eq_col], axis=1)
    is_equal_x1 = df.apply(lambda x: x["lst_eq_hyp"][0] == x[eq_col][0], axis=1)
    is_equal_x0 = df.apply(lambda x: x["lst_eq_hyp"][1] == x[eq_col][1], axis=1)

    acc_overall = np.mean(is_equal)
    acc_x1 = np.mean(is_equal_x1)
    acc_x0 = np.mean(is_equal_x0)
    if "correct" in df.columns:
        answer_is_correct = df["correct"]
    else:
        answer_is_correct = pd.Series(dtype=bool)

    # null values - model declined to answer
    null_values = df["lst_eq_hyp"].apply(lambda x: None in x)
    null_x0 = df["lst_eq_hyp"].apply(lambda x: x[1] is None)
    null_x1 = df["lst_eq_hyp"].apply(lambda x: x[0] is None)
    num_declined = np.sum(null_values)
    num_declined_x0 = np.sum(null_x0)
    num_declined_x1 = np.sum(null_x1)
    print("Number of declined answers: ", num_declined)
    print("Number of declined answers x0: ", num_declined_x0)
    print("Number of declined answers x1: ", num_declined_x1)

    # print out inputs of declined answers?
    print("Inputs of declined answers: ")
    df[null_values].to_csv("inputs_of_declined_answers.csv")

    # rmse_x1 = np.sqrt(np.mean((df["lst_eq_hyp"].apply(lambda x: x[0]) - df[eq_col].apply(lambda x: x[0])) ** 2))
    # rmse_x0 = np.sqrt(np.mean((df["lst_eq_hyp"].apply(lambda x: x[1]) - df[eq_col].apply(lambda x: x[1])) ** 2))

    median_squared_error_x1 = np.median(
        (df["lst_eq_hyp"].apply(lambda x: x[0]) - df[eq_col].apply(lambda x: x[0])) ** 2
    )
    median_squared_error_x0 = np.median(
        (df["lst_eq_hyp"].apply(lambda x: x[1]) - df[eq_col].apply(lambda x: x[1])) ** 2
    )

    return {
        "acc_overall": acc_overall,
        "acc_x1": acc_x1,
        "acc_x0": acc_x0,
        "answer_correct": answer_is_correct.tolist(),
        "is_equal": is_equal.tolist(),
        "rmse_x1": median_squared_error_x1,
        "rmse_x0": median_squared_error_x0,
        "x1_induced": df["lst_eq_hyp"].apply(lambda x: x[0]).tolist(),
        "x0_induced": df["lst_eq_hyp"].apply(lambda x: x[1]).tolist(),
    }
