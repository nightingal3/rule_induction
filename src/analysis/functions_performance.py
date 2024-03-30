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
import pathlib
import os

from src.utils.parsing_utils import parse_polynomial, equation_accuracy

sns.set_theme(font_scale=1.3, style="white")


def parse_numeric_answer(answer: Union[str, int]) -> int:
    # remove "output: " from the beginning of the string if it exists
    if isinstance(answer, int) or isinstance(answer, float):
        return answer

    answer = answer.lower()
    answer = answer[8:] if answer[:8] == "output: " else answer
    if "\n" in answer:
        answer = answer.split("\n")[0]
    # convert to int
    try:
        answer = int(answer)
    except:
        return None
    return answer


def parse_function_outputs(output_text: str, prompt_style: str = "base"):
    # Normalize line breaks and lowercase
    output_text = output_text.strip()
    output_text = output_text.replace("\r\n", "\n").lower()
    if prompt_style == "zs-cot":
        pattern = r"final output: (.*)"

        match = re.search(pattern, output_text)

        if match:
            valid_output = match.group(1)
        else:
            # try looking for just "output: " and then the number
            pattern = r"output: (.*)"
            match = re.search(pattern, output_text)
            if match:
                valid_output = match.group(1)
            else:
                return None

        return valid_output
    else:
        # Pattern to find "Output:" and the output number
        pattern = r"output:\s*(-?\d+)"
        matches = list(re.finditer(pattern, output_text, re.IGNORECASE | re.DOTALL))

        valid_output = None
        for match in matches:
            # Extract the context for this "Output:"
            start_pos = match.start()
            context = output_text[:start_pos]
            context_lines = context.split("\n")

            # Check if the context does not contain "Input:" immediately before this "Output:"
            if len(context_lines) < 2 or "input:" not in context_lines[-2]:
                valid_output = match.group(1)

        if valid_output is not None:
            try:
                return int(valid_output.strip())
            except ValueError:
                return None
        else:
            return None


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
    agreement_measure: str = "cohen_kappa",
) -> float:
    if agreement_type == "base":
        correct_vec = df_base["correct"].astype(int).tolist()
    else:
        correct_vec = df_rule["correct"].astype(int).tolist()
    rule_correctness_vec = np.array(rule_correctness_vec).astype(int)
    if agreement_measure == "cohen_kappa":
        return cohen_kappa_score(rule_correctness_vec, correct_vec)
    else:  # point-biserial
        return pointbiserialr(rule_correctness_vec, correct_vec)


def plot_true_vs_predicted_answer(
    ax: plt.Axes,
    df: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str,
    color: str = "blue",
) -> None:
    sns.scatterplot(x=y_true_col, y=y_pred_col, data=df, color=color, ax=ax)
    ax.set_xlim([-450, 450])


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


def plot_regression_line(ax, x, y, color):
    # Fit the regression line
    x = np.array(x)
    y = np.array(y)
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]

    # Plot the regression line
    ax.plot(x, m * x + c, color=color)


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
    axs[0].set_title("x0 coeff")
    axs[1].scatter(real_coefficients_x1, predicted_x1, label="x1", color="orange")
    axs[1].set_title("x1 coeff")
    plot_regression_line(axs[0], real_coefficients_x0, predicted_x0, color="blue")
    plot_regression_line(axs[1], real_coefficients_x1, predicted_x1, color="orange")

    axs[0].set_xlabel("Real coefficients")
    axs[0].set_ylabel("Predicted coefficients")
    # plt.title(f"Predicted vs real coefficients - {model} (null values excluded)")

    # limit to range -20, 20 on y (remove this after)
    axs[0].set_ylim([-20, 20])
    axs[1].set_ylim([-20, 20])

    axs[0].set_aspect("equal", "box")
    axs[1].set_aspect("equal", "box")

    split_low = -20
    split_high = 20

    print("Correlation: ")
    spearman_corr_x0, _ = spearmanr(real_coefficients_x0, predicted_x0)
    spearman_corr_x1, _ = spearmanr(real_coefficients_x1, predicted_x1)

    print("x0: ", spearmanr(real_coefficients_x0, predicted_x0))
    print("x1: ", spearmanr(real_coefficients_x1, predicted_x1))

    x0_annotate_pos = (
        min(axs[0].get_xlim()) + np.ptp(axs[0].get_xlim()) * 0.05,
        max(axs[0].get_ylim()) - np.ptp(axs[0].get_ylim()) * 0.1,
    )
    x1_annotate_pos = (
        min(axs[1].get_xlim()) + np.ptp(axs[1].get_xlim()) * 0.05,
        max(axs[1].get_ylim()) - np.ptp(axs[1].get_ylim()) * 0.1,
    )

    axs[0].annotate(
        f"Spearman r: {spearman_corr_x0:.2f}",
        xy=x0_annotate_pos,
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1),
    )

    axs[1].annotate(
        f"Spearman r: {spearman_corr_x1:.2f}",
        xy=x1_annotate_pos,
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1),
    )

    plt.tight_layout()

    plt.savefig(f"./src/analysis/figures/{output_name}.png")
    plt.savefig(f"./src/analysis/figures/{output_name}.pdf")


if __name__ == "__main__":
    base_dir = pathlib.Path(
        "/data/tir/projects/tir5/users/mengyan3/rule_induction/logs"
    )

    pattern = "**/*/*/*/*/*.csv"

    # parser = argparse.ArgumentParser()

    # parser.add_argument("--model_name", default="gpt-3.5-turbo", type=str)
    # parser.add_argument("--setting", type=str, default="base")

    # args = parser.parse_args()

    model_names = ["gpt-3.5-turbo", "gpt-4-turbo", "Llama-2-7b", "Llama-2-70b"]
    settings = ["base", "grammar_induction", "full_grammar"]
    fig, axs = plt.subplots(4, 3, figsize=(15, 20))

    for i, model_name in enumerate(model_names):
        for j, setting in enumerate(settings):
            df_concat = None
            for file_path in base_dir.glob(pattern):
                if not str(file_path).endswith(".csv"):
                    continue
                if not "functions" in str(file_path):
                    continue
                if "HYPSFILE" in str(file_path):
                    continue
                if not model_name in str(file_path):
                    continue
                if not setting in str(file_path):
                    continue

                df = pd.read_csv(file_path)
                # breakpoint()
                df["outputs"] = [
                    parse_function_outputs(x, setting) for x in df["outputs"]
                ]
                # breakpoint()
                df = df.dropna(subset=["outputs"])
                if df_concat is None:
                    df_concat = df
                else:
                    df_concat = pd.concat([df_concat, df], ignore_index=True)

            if setting == "full_grammar":
                color = "#DB69B6"
            elif setting == "base":
                color = "#DBB36B"
            else:
                color = "#699BDB"
            plot_true_vs_predicted_answer(
                axs[i, j],
                df_concat,
                "outputs",
                "answer",
                color=color,
            )

            # if setting == "grammar_induction":
            #     df_concat["equation_hypothesis"] = df_concat["inputs"].apply(
            #         parse_polynomial
            #     )
            #     eq_acc = equation_accuracy(
            #         df_concat, "equation_hypothesis", "task_id", 1
            #     )
            #     print("Equation accuracy: ")
            #     print(eq_acc)

            #     print("Rule correctness vs accuracy: ")
            #     rule_is_correct = eq_acc["is_equal"]
            #     print(
            #         "Agreement (base): ",
            #         agreement_rule_acc_correctness(
            #             df_induced_grammar,
            #             df_base,
            #             "outputs",
            #             "task_id",
            #             rule_is_correct,
            #             agreement_type="base",
            #         ),
            #     )
            #     print(
            #         "Agreement (in-context rule): ",
            #         agreement_rule_acc_correctness(
            #             df_induced_grammar,
            #             df_base,
            #             "outputs",
            #             "task_id",
            #             rule_is_correct,
            #             agreement_type="in_context",
            #         ),
            #     )

            #     x0_true, x1_true = df_induced_grammar["task_id"].apply(
            #         lambda x: x[0]
            #     ), df_induced_grammar["task_id"].apply(lambda x: x[1])
            #     distance_x0 = np.abs(np.array(x0_true) - np.array(eq_acc["x0_induced"]))
            #     distance_x1 = np.abs(np.array(x1_true) - np.array(eq_acc["x1_induced"]))

            # Adjust layout to prevent overlapping
    plt.tight_layout()

    # Save the entire figure
    plt.savefig("./src/analysis/all_subfigures.png")
    plt.savefig("./src/analysis/all_subfigures.pdf")
    assert False
    df_base = pd.read_csv(
        f"./logs/functions/functions_simple_base_{args.model}_None_None_minset_False_loop_False_temp_0.0_few_shot_examples_False_degree_1.csv"
    )
    # df_induced_grammar = pd.read_csv(
    # f"./logs/functions/functions_simple_grammar_induction_{args.model}_minset_False_loop_False_temp_0.0_few_shot_examples_False_start_0_end_200_degree_1.csv"
    # )

    df_induced_grammar = pd.read_csv(
        "/data/tir/projects/tir5/users/mengyan3/rule_induction/logs/functions/p_answer_given_hyp_logprobs/run_1/functions_simple_grammar_induction_gpt-3.5-turbo_minset_False_loop_False_temp_0.0_few_shot_examples_False_start_0_end_200_degree_1_num_hyps_5_p_answer_given_hyp_logprobs.csv"
    )

    df_induced_grammar["equation_hypothesis"] = df_induced_grammar["inputs"].apply(
        parse_polynomial
    )

    df_induced_grammar.to_csv(f"df_induced_grammar_{args.model}.csv", index=False)
    df_full_grammar = pd.read_csv(
        f"/data/tir/projects/tir5/users/mengyan3/rule_induction/logs/functions/functions_simple_full_grammar_{args.model}_None_None_minset_False_loop_False_temp_0.0_few_shot_examples_False_degree_1.csv"
    )

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
