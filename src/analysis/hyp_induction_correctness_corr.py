import pathlib
import argparse
import matplotlib.pyplot as plt
import pandas as pd

from functions_performance import parse_function_outputs, agreement_rule_acc_correctness
from src.utils.parsing_utils import parse_polynomial, equation_accuracy

if __name__ == "__main__":
    base_dir = pathlib.Path(
        "/data/tir/projects/tir5/users/mengyan3/rule_induction/logs"
    )

    pattern = "**/*/*/*/*/*.csv"

    hyps_methods = [
        "ground_truth",
        # "p_data_given_hyp_logprobs",
        # "p_answer_given_hyp_logprobs",
        # "p_data_given_hyp_guess",
    ]

    model_names = [
        "gpt-3.5-turbo",
        "gpt-4-turbo",
        # "meta-llama/Llama-2-70b-chat-hf",
        # "meta-llama/Llama-2-7b-chat-hf",
    ]

    hyps_methods_for_df = []
    x0_correctness = []
    x1_correctness = []
    agreement_base = []
    agreement_in_context = []
    pvals_base = []
    pvals_in_context = []
    model_names_for_df = []
    for hyps_method in hyps_methods:
        for i, model_name in enumerate(model_names):
            print("MODEL: ", model_name)
            base_df = None
            hyps_df = None
            curr_type = None
            for file_path in base_dir.glob(pattern):
                if not str(file_path).endswith(".csv"):
                    continue
                if not "functions" in str(file_path):
                    continue
                if not model_name in str(file_path):
                    continue
                if "HYPSFILE" in str(file_path):
                    continue
                if not "base" in str(file_path) and not "grammar_induction" in str(
                    file_path
                ):
                    continue

                if "base" in str(file_path):
                    print("BASE: ", file_path)
                    if base_df is None:
                        base_df = pd.read_csv(file_path)
                    else:
                        base_df = pd.concat([base_df, pd.read_csv(file_path)])
                if "grammar_induction" in str(file_path) and hyps_method in str(
                    file_path
                ):
                    print("HYPS: ", file_path)
                    if hyps_df is None:
                        hyps_df = pd.read_csv(file_path)
                    else:
                        hyps_df = pd.concat([hyps_df, pd.read_csv(file_path)])

            if base_df is not None and hyps_df is not None:
                if model_name == "Llama-2-7b":
                    breakpoint()
                hyps_df["equation_hypothesis"] = hyps_df["inputs"].apply(
                    parse_polynomial
                )
                eq_acc = equation_accuracy(hyps_df, "equation_hypothesis", "task_id", 1)
                rule_is_correct = eq_acc["is_equal"]
                print("x0 correctness: ", eq_acc["acc_x0"])
                print("x1 correctness: ", eq_acc["acc_x1"])

                agreement_with_base, pvalue_base = agreement_rule_acc_correctness(
                    hyps_df,
                    base_df,
                    "outputs",
                    "task_id",
                    rule_is_correct,
                    agreement_type="base",
                    agreement_measure="pointbiserial",
                )

                agreement_with_instruction, pvalue_instruction = (
                    agreement_rule_acc_correctness(
                        hyps_df,
                        base_df,
                        "outputs",
                        "task_id",
                        rule_is_correct,
                        agreement_type="in_context",
                        agreement_measure="pointbiserial",
                    )
                )

                print("Agreement (base): ", agreement_with_base)
                print("p-value (base): ", pvalue_base)
                print("Agreement (in-context rule): ", agreement_with_instruction)
                print("p-value (in-context rule): ", pvalue_instruction)

                hyps_methods_for_df.append(hyps_method)
                x0_correctness.append(eq_acc["acc_x0"])
                x1_correctness.append(eq_acc["acc_x1"])
                agreement_base.append(agreement_with_base)
                pvals_base.append(pvalue_base)
                agreement_in_context.append(agreement_with_instruction)
                pvals_in_context.append(pvalue_instruction)

                model_names_for_df.append(model_name)

    df = pd.DataFrame(
        {
            "hyps_method": hyps_methods_for_df,
            "x0_correctness": x0_correctness,
            "x1_correctness": x1_correctness,
            "agreement_base": agreement_base,
            "pvals_base": pvals_base,
            "agreement_in_context": agreement_in_context,
            "pvals_in_context": pvals_in_context,
            "model_name": model_names_for_df,
        }
    )

    print(df)
