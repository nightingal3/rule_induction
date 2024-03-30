import argparse
import pandas as pd
import pathlib

from src.utils.parsing_utils import parse_polynomial, equation_accuracy
from src.analysis.functions_performance import plot_predicted_vs_real_coefficients

pattern = "**/*/*/*/*/*.csv"
base_dir = pathlib.Path("/data/tir/projects/tir5/users/mengyan3/rule_induction/logs")
rank_0_only = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--setting", type=str, default="grammar_induction")
    args = parser.parse_args()
    # parser.add_argument("--hyps_file", type=str, required=True)
    # args = parser.parse_args()
    all_real_x0s = []
    all_real_x1s = []
    all_predicted_x0s = []
    all_predicted_x1s = []
    setting = []
    num_refused_total = 0
    num_refused_x0_total = 0
    num_refused_x1_total = 0

    for file_path in base_dir.glob(pattern):
        if not str(file_path).endswith("HYPSFILE.csv"):
            continue
        if "functions" not in str(file_path):
            continue
        if "grammar_induction" not in str(file_path):
            continue
        if args.model_name not in str(file_path):
            continue

        print("File path", file_path)
        df_hyps = pd.read_csv(str(file_path))
        if rank_0_only:
            # only keep rank 0 hypotheses
            df_hyps = df_hyps[df_hyps["rank"] == 1]

        if len(df_hyps) < 1000:  # should be 1k hypotheses
            continue
        df_sorted = df_hyps.sort_values(
            by=["task_id", "all_probs"], ascending=[True, False]
        )
        df_sorted["rank"] = df_sorted.groupby("task_id")["all_probs"].rank(
            "dense", ascending=False
        )

        df_sorted["parsed_equations"] = df_sorted["all_hyps"].apply(
            lambda x: parse_polynomial(x)
        )

        # rank vs correctness
        results = equation_accuracy(df_sorted, "parsed_equations", "task_id", 1)

        df_sorted["eq_correct"] = results["is_equal"]
        real_x0s = df_sorted["task_id"].apply(lambda x: x[1])
        real_x1s = df_sorted["task_id"].apply(lambda x: x[0])

        predicted_x0s = df_sorted["lst_eq_hyp"].apply(lambda x: x[1])
        predicted_x1s = df_sorted["lst_eq_hyp"].apply(lambda x: x[0])

        all_real_x0s.extend(real_x0s)
        all_real_x1s.extend(real_x1s)
        all_predicted_x0s.extend(predicted_x0s)
        all_predicted_x1s.extend(predicted_x1s)
        num_refused_total += results["num_refused"]
        num_refused_x0_total += results["num_refused_x0"]
        num_refused_x1_total += results["num_refused_x1"]

    plot_predicted_vs_real_coefficients(
        f"{args.model_name}",
        real_x0s,
        real_x1s,
        predicted_x0s,
        predicted_x1s,
        f"{args.model_name}_{args.setting}_predicted_vs_real_coefficients",
    )
    print("Total num refused", num_refused_total)
    print("Total num refused x0", num_refused_x0_total)
    print("Total num refused x1", num_refused_x1_total)
    print("TOTAL HYPOTHESIS COUNT: ", len(all_real_x0s))

    assert False
    plot_predicted_vs_real_coefficients(
        "gpt-4-turbo",
        real_x0s,
        real_x1s,
        predicted_x0s,
        predicted_x1s,
        "predicted_vs_real_after_pdata_logprobs",
    )

    print("x0 accuracy")
    print(results["acc_x0"])
    print("x1 accuracy")
    print(results["acc_x1"])
    print("exact match")
    print(results["acc_overall"])
