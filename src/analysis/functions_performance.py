import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, List, Callable
import re
import ast
import argparse
import numpy as np
from sklearn.metrics import cohen_kappa_score
from scipy.stats import pearsonr, spearmanr, pointbiserialr

from src.utils.utils import parse_polynomial, equation_accuracy


def parse_numeric_answer(answer: Union[str, int]) -> int:
    # remove "output: " from the beginning of the string if it exists
    if isinstance(answer, int) or isinstance(answer, float):
        return answer

    answer = answer.lower()
    answer = answer[8:] if answer[:8] == "output: " else answer
    # convert to int
    try:
        answer = int(answer)
    except:
        return None
    return answer


def mse(df: pd.DataFrame, y_true_col: str, y_pred_col: str) -> float:
    df[y_pred_col] = df[y_pred_col].apply(parse_numeric_answer)
    df[y_true_col] = df[y_true_col].apply(parse_numeric_answer)

    df = df.dropna(subset=[y_pred_col, y_true_col])

    return np.mean((df[y_true_col] - df[y_pred_col]) ** 2)


def median_squared_error(df: pd.DataFrame, y_true_col: str, y_pred_col: str) -> float:
    df[y_pred_col] = df[y_pred_col].apply(parse_numeric_answer)
    df[y_true_col] = df[y_true_col].apply(parse_numeric_answer)

    df = df.dropna(subset=[y_pred_col, y_true_col])
    return np.median((df[y_true_col] - df[y_pred_col]) ** 2)


def accuracy(df: pd.DataFrame, is_correct_col: str) -> float:
    return np.mean(df[is_correct_col])


def mse_by_function_id(
    df: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str,
    function_id_col: str,
    func: Callable,
) -> float:
    # convert list col back to str if it is a list
    if isinstance(df[function_id_col].iloc[0], list):
        df[function_id_col] = df[function_id_col].apply(lambda x: tuple(x))

    return df.groupby(function_id_col).apply(lambda x: func(x, y_true_col, y_pred_col))


def accuracy_by_function_id(
    df: pd.DataFrame, is_correct_col: str, function_id_col: str
) -> float:
    # convert list col back to str if it is a list
    if isinstance(df[function_id_col].iloc[0], list):
        df[function_id_col] = df[function_id_col].apply(lambda x: tuple(x))

    return df.groupby(function_id_col).apply(lambda x: accuracy(x, is_correct_col))


def rule_accuracy(
    df: pd.DataFrame, model_outputs_col: str, ground_truth_col: str
) -> float:
    # parse the output for
    return np.mean(df[model_outputs_col] == df[ground_truth_col])


def agreement_rule_acc_correctness(
    df_rule: pd.DataFrame,
    df_base: pd.DataFrame,
    model_outputs_col: str,
    ground_truth_col: str,
    rule_correctness_vec: list,
    agreement_type: str = "base",
) -> float:
    if agreement_type == "base":
        correct_vec = df_base["correct"].astype(int).tolist()
    else:
        correct_vec = df_rule["correct"].astype(int).tolist()
    rule_correctness_vec = np.array(rule_correctness_vec).astype(int)
    return cohen_kappa_score(rule_correctness_vec, correct_vec)


def plot_true_vs_predicted_answer(
    df: pd.DataFrame, y_true_col: str, y_pred_col: str, out_name: str
) -> None:
    df[y_pred_col] = df[y_pred_col].apply(parse_numeric_answer)
    df[y_true_col] = df[y_true_col].apply(parse_numeric_answer)
    plt.figure(figsize=(10, 10))
    sns.scatterplot(x=y_true_col, y=y_pred_col, data=df)
    plt.savefig(f"{out_name}.png")
    plt.savefig(f"{out_name}.pdf")


def plot_accuracy_per_coefficient(
    df_base: pd.DataFrame, df_full_grammar: pd.DataFrame, out_name: str
) -> None:
    acc_by_function_id_base = accuracy_by_function_id(df_base, "correct", "task_id")
    acc_by_function_id_full_grammar = accuracy_by_function_id(
        df_full_grammar, "correct", "task_id"
    )
    # create a grouped bar plot with the x axis being the function (arranged by sum of two coefficients), and y axis being accuracy
    # first, sort the functions by the sum of the two coefficients
    df_base["sum_of_coefficients"] = df_base["coefficients"].apply(
        lambda x: sum([abs(float(c)) for c in x.split(" ")])
    )
    df_full_grammar["sum_of_coefficients"] = df_full_grammar["coefficients"].apply(
        lambda x: sum([abs(float(c)) for c in x.split(" ")])
    )
    # order task_id by the sum of the coefficients
    df_base = df_base.sort_values(by="sum_of_coefficients")
    df_full_grammar = df_full_grammar.sort_values(by="sum_of_coefficients")
    plt.figure(figsize=(10, 10))
    sns.barplot(
        x="task_id",
        y="correct",
        data=df_base,
        order=df_base["task_id"],
        color="blue",
        label="Base",
    )
    sns.barplot(
        x="task_id",
        y="correct",
        data=df_full_grammar,
        order=df_full_grammar["task_id"],
        color="red",
        label="Full grammar",
    )
    plt.legend()
    plt.savefig(f"{out_name}.png")


def plot_predicted_vs_real_coefficients(
    model: str,
    real_coefficients_x0: List,
    real_coefficients_x1: List,
    predicted_x0: List,
    predicted_x1: List,
    output_name: str = "predicted_vs_real_coefficients",
    exclude_outliers: bool = False,
) -> None:
    # subplots
    fig, axs = plt.subplots(1, 2)
    # exclude null values and only plot the ones where the model predicted something
    valid_indices_x0 = [
        i
        for i in range(len(real_coefficients_x0))
        if predicted_x0[i] is not None and not np.isnan(predicted_x0[i])
    ]
    valid_indices_x1 = [
        i
        for i in range(len(real_coefficients_x1))
        if predicted_x1[i] is not None and not np.isnan(predicted_x1[i])
    ]

    real_coefficients_x0 = [real_coefficients_x0[i] for i in valid_indices_x0]
    real_coefficients_x1 = [real_coefficients_x1[i] for i in valid_indices_x1]
    predicted_x0 = [predicted_x0[i] for i in valid_indices_x0]
    predicted_x1 = [predicted_x1[i] for i in valid_indices_x1]

    axs[0].scatter(real_coefficients_x0, predicted_x0, label="x0")
    axs[0].set_title("x0")
    axs[1].scatter(real_coefficients_x1, predicted_x1, label="x1", color="orange")
    axs[1].set_title("x1")
    fig.legend()
    plt.xlabel("Real coefficients")
    plt.ylabel("Predicted coefficients")
    plt.title(f"Predicted vs real coefficients - {model} (null values excluded)")

    # limit to range -20, 20 on y (remove this after)
    # axs[0].set_ylim([-20, 20])
    # axs[1].set_ylim([-20, 20])

    plt.tight_layout()

    print("Correlation: ")
    print("x0: ", spearmanr(real_coefficients_x0, predicted_x0))
    print("x1: ", spearmanr(real_coefficients_x1, predicted_x1))

    plt.savefig(f"{output_name}.png")
    plt.savefig(f"{output_name}.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-3.5-turbo", type=str)
    args = parser.parse_args()

    df_base = pd.read_csv(
        f"./logs/functions/functions_simple_base_{args.model}_None_None_minset_False_loop_False_temp_0.0_few_shot_examples_False_degree_1.csv"
    )
    # df_induced_grammar = pd.read_csv(
    # f"./logs/functions/functions_simple_grammar_induction_{args.model}_minset_False_loop_False_temp_0.0_few_shot_examples_False_start_0_end_200_degree_1.csv"
    # )

    df_induced_grammar = pd.read_csv(
        "logs/functions/p_data_given_hyp_logprobs/run_1/functions_simple_grammar_induction_gpt-3.5-turbo_minset_False_loop_False_temp_0.0_few_shot_examples_False_start_0_end_200_degree_1_num_hyps_5_p_data_given_hyp_logprobs.csv"
    )

    df_induced_grammar["equation_hypothesis"] = df_induced_grammar["inputs"].apply(
        parse_polynomial
    )

    df_induced_grammar.to_csv(f"df_induced_grammar_{args.model}.csv", index=False)
    df_full_grammar = pd.read_csv(
        f"/data/tir/projects/tir5/users/mengyan3/rule_induction/logs/functions/functions_simple_full_grammar_{args.model}_None_None_minset_False_loop_False_temp_0.0_few_shot_examples_False_degree_1.csv"
    )
    eq_acc = equation_accuracy(df_induced_grammar, "equation_hypothesis", "task_id", 1)
    print("Equation accuracy: ")
    print(eq_acc)

    print("Rule correctness vs accuracy: ")
    rule_is_correct = eq_acc["is_equal"]
    print(
        "Agreement (base): ",
        agreement_rule_acc_correctness(
            df_induced_grammar,
            df_base,
            "outputs",
            "task_id",
            rule_is_correct,
            agreement_type="base",
        ),
    )
    print(
        "Agreement (in-context rule): ",
        agreement_rule_acc_correctness(
            df_induced_grammar,
            df_base,
            "outputs",
            "task_id",
            rule_is_correct,
            agreement_type="in_context",
        ),
    )

    x0_true, x1_true = df_induced_grammar["task_id"].apply(
        lambda x: x[0]
    ), df_induced_grammar["task_id"].apply(lambda x: x[1])
    distance_x0 = np.abs(np.array(x0_true) - np.array(eq_acc["x0_induced"]))
    distance_x1 = np.abs(np.array(x1_true) - np.array(eq_acc["x1_induced"]))

    # filter out infs
    # print(pointbiserialr(eq_acc["answer_correct"], distance_x0 + distance_x1))
    # print(pointbiserialr(eq_acc["answer_correct"], distance_x0 + distance_x1))

    plot_predicted_vs_real_coefficients(
        args.model,
        x0_true,
        x1_true,
        eq_acc["x0_induced"],
        eq_acc["x1_induced"],
        f"predicted_vs_real_coefficients_{args.model}_guess",
    )

    print("ACCURACY: ")
    print("Base: ", accuracy(df_base, "correct"))
    print("Full grammar: ", accuracy(df_full_grammar, "correct"))
    print("Induced grammar: ", accuracy(df_induced_grammar, "correct"))

    print("median squared error: ")
    print("Base: ", median_squared_error(df_base, "outputs", "answer"))
    print("Full grammar: ", median_squared_error(df_full_grammar, "outputs", "answer"))
    print(
        "Induced grammar: ",
        median_squared_error(df_induced_grammar, "outputs", "answer"),
    )

    print("median squared error by function id: ")
    print(
        "Base: ",
        mse_by_function_id(
            df_base, "outputs", "answer", "task_id", median_squared_error
        ),
    )
    print(
        "Full grammar: ",
        mse_by_function_id(
            df_full_grammar, "outputs", "answer", "task_id", median_squared_error
        ),
    )

    print("Accuracy by function id: ")
    print("Base: ", accuracy_by_function_id(df_base, "correct", "task_id"))
    print(
        "Full grammar: ", accuracy_by_function_id(df_full_grammar, "correct", "task_id")
    )

    plot_true_vs_predicted_answer(
        df_base, "outputs", "answer", f"base_true_vs_predicted_{args.model}"
    )
    plot_true_vs_predicted_answer(
        df_full_grammar,
        "outputs",
        "answer",
        f"full_grammar_true_vs_predicted_{args.model}_guess",
    )
