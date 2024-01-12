import re
import numpy as np
import pandas as pd
import ast


def parse_brackets(polynomial: str) -> str:
    while "(" in polynomial:
        start = polynomial.rfind("(")
        end = polynomial.find(")", start)
        if end == -1:
            raise ValueError(f"Unbalanced brackets in polynomial {polynomial}")

        bracketed_expr = polynomial[start + 1 : end]
        bracketed_expr = bracketed_expr.replace("+-", "-")
        polynomial = polynomial[:start] + bracketed_expr + polynomial[end + 1 :]

    return polynomial


def isempty_match(match):
    return all([(group == "") or (group is None) for group in match.groups()])


def parse_polynomial(orig_polynomial: str) -> dict:
    # This function was written by chatgpt and postedited by me. Use at own risk but seems to be working
    # Remove spaces and 'y=' part if present
    # model outputs this when it declines to answer
    # if orig_polynomial == "y = -x - 1":
    # breakpoint()

    if "y" in orig_polynomial:
        polynomial = orig_polynomial.split("y =")[-1].split("\n")[0]
    else:
        if "Output" in orig_polynomial:
            polynomial = orig_polynomial.split("Output =")[-1].split("\n")[0]
        else:
            return {0: None, 1: None}

    if "/" in polynomial:  # reject unsimplified fractions
        return {}

    polynomial = polynomial.replace("-", "+-")
    polynomial = parse_brackets(polynomial)
    polynomial = polynomial.replace("--", "")
    polynomial = polynomial.replace("- -", "")
    polynomial = polynomial.replace(" ", "")

    # Split the polynomial into terms
    terms = polynomial.split("+")
    terms = [term.strip() for term in terms if term != ""]

    # Regular expression to match coefficients and powers
    # regex = r"(-?\d*)(x\^?(\d*))?"
    # regex = r"(-?\d*\.?\d*)(x\^?(\d*))?"
    regex = r"(-?\d*\.?\d*|a|b|c)(x\^?(\d*))?"

    # Dictionary to store coefficients with their degrees
    coefficients = {}

    for term in terms:
        match = re.match(regex, term)
        if match and not isempty_match(match):
            # Extract coefficient and power
            coeff_str, _, power_str = match.groups()

            # exclude "non-answers" like y = ax + b
            if coeff_str and re.fullmatch(r"[a-z]+", coeff_str):
                return {}

            # Set default values if missing
            try:
                if coeff_str == "-":
                    coeff = -1
                else:
                    coeff = float(coeff_str) if coeff_str else 1
                power = int(power_str) if power_str else 1 if "x" in term else 0
            except:
                print("=== Error parsing: ", orig_polynomial)
                # if we can't parse the polynomial, return None
                breakpoint()
                return {}

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
