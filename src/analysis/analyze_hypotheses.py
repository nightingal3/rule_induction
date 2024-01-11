import argparse
import pandas as pd

from src.utils.utils import parse_polynomial, equation_accuracy
from src.analysis.functions_performance import plot_predicted_vs_real_coefficients


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hyps_file", type=str, required=True)
    args = parser.parse_args()

    df_hyps = pd.read_csv(args.hyps_file)
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

    plot_predicted_vs_real_coefficients(
        "gpt-3.5-turbo",
        real_x0s,
        real_x1s,
        predicted_x0s,
        predicted_x1s,
        "predicted_vs_real_after_pdata_guess",
    )

    print("x0 accuracy")
    print(results["acc_x0"])
    print("x1 accuracy")
    print(results["acc_x1"])
    print("exact match")
    print(results["acc_overall"])
