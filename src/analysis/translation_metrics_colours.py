import pandas as pd
import evaluate
import matplotlib.pyplot as plt
import argparse
import pathlib
import seaborn as sns

chrf_metric = evaluate.load("chrf")
bleu_metric = evaluate.load("bleu")

sns.set_theme(font_scale=1.35, style="white")


color_palette_methods = {
    "few-shot": "#DBB36B",
    "zs-cot": "#77DB69",
    "true-instruction": "#DB69B6",
    "instruction_induction": "#699BDB",
}

old_to_new_names = {
    "Base": "few-shot",
    "Full Grammar": "true-instruction",
    "Grammar Induction": "instruction_induction",
}


def split_and_calc_metrics(df, runs=6):
    # Assuming each run has an equal number of observations
    run_length = len(df) // runs
    scores = []

    for i in range(runs):
        # Splitting the dataframe into individual runs
        run_df = df.iloc[i * run_length : (i + 1) * run_length]
        # Calculate metrics for the run
        chrf_score = chrf_metric.compute(
            predictions=run_df["outputs"]
            .str.replace("Output: ", "", regex=False)
            .tolist(),
            references=run_df["answer"].tolist(),
        )
        scores.append(chrf_score["score"])

    return scores


def calc_metrics(df):
    chrf = chrf_metric.compute(
        predictions=df["outputs"].tolist(), references=df["answer"].tolist()
    )
    bleu = bleu_metric.compute(
        predictions=df["outputs"].tolist(), references=df["answer"].tolist()
    )
    return chrf, bleu


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    data = {}
    args = parser.parse_args()
    nonce_words = ["lug", "dax", "wif", "zup", "bluf", "walm"]

    pattern = "**/*/*/*/*/*.csv"
    base_dir = pathlib.Path(
        "/data/tir/projects/tir5/users/mengyan3/rule_induction/logs"
    )
    models = [
        "gpt-3.5-turbo",
        "gpt-4-turbo",
        "Llama-2-7b-chat-hf",
        "Llama-2-70b-chat-hf",
    ]

    model_names_for_plot = []
    methods_for_plot = []
    chrfs_for_plot = []
    results = []

    for model_name in models:
        base_df = None
        grammar_induction_df = None
        ground_truth_df = None

        for file_path in base_dir.glob(pattern):
            if str(file_path).endswith("HYPSFILE.csv"):
                continue
            if "colours" not in str(file_path):
                continue
            if model_name not in str(file_path):
                continue

            df = pd.read_csv(file_path)
            if len(df) < 200:
                continue
            df["outputs"] = df["outputs"].str.replace("Output: ", "", regex=False)

            if "base" in str(file_path):
                print("BASE: ", file_path)
                if base_df is None:
                    base_df = df
                else:
                    base_df = pd.concat([base_df, df])
            elif "grammar_induction" in str(file_path):
                print("GRAMMAR INDUCTION: ", file_path)
                if grammar_induction_df is None:
                    grammar_induction_df = df
                else:
                    grammar_induction_df = pd.concat([grammar_induction_df, df])
            elif "full_grammar" in str(file_path):
                print("GROUND TRUTH: ", file_path)
                if ground_truth_df is None:
                    ground_truth_df = pd.read_csv(file_path)
                else:
                    ground_truth_df = pd.concat(
                        [ground_truth_df, pd.read_csv(file_path)]
                    )

        base_metrics_per_run = split_and_calc_metrics(base_df)
        grammar_induction_metrics_per_run = split_and_calc_metrics(grammar_induction_df)
        ground_truth_metrics_per_run = split_and_calc_metrics(ground_truth_df)

        for i in range(6):
            base_metrics = base_metrics_per_run[i]
            grammar_induction_metrics = grammar_induction_metrics_per_run[i]
            ground_truth_metrics = ground_truth_metrics_per_run[i]

            model_names_for_plot.extend([model_name] * 3)
            methods_for_plot.extend(["Base", "Grammar Induction", "Full Grammar"])
            chrfs_for_plot.extend(
                [
                    s
                    for s in [
                        base_metrics,
                        grammar_induction_metrics,
                        ground_truth_metrics,
                    ]
                ]
            )

            for nonce_word in nonce_words:
                for method, df in zip(
                    ["base", "grammar_induction", "full_grammar"],
                    [base_df, grammar_induction_df, ground_truth_df],
                ):
                    if df is not None:
                        # Filter examples containing the nonce word
                        nonce_df = df[
                            df["inputs"].str.contains(rf"\b{nonce_word}\b", regex=True)
                        ]
                        chrf_mean, bleu_mean = calc_metrics(nonce_df)
                        chrf_mean = chrf_mean["score"]
                        bleu_mean = bleu_mean["bleu"]
                        # Append the results
                        results.append(
                            {
                                "model": model_name,
                                "method": method,
                                "nonce_word": nonce_word,
                                "chrf": chrf_mean,
                                "bleu": bleu_mean,
                            }
                        )

    final_df = pd.DataFrame(
        {
            "model_name": model_names_for_plot,
            "method": methods_for_plot,
            "chrf": chrfs_for_plot,
        }
    )
    final_df["method"] = final_df["method"].replace(old_to_new_names)

    g = sns.catplot(
        data=final_df,
        x="model_name",
        y="chrf",
        hue="method",
        kind="bar",
        palette=color_palette_methods,
    )
    plt.xticks(rotation=45)
    g.figure.subplots_adjust(right=0.67)  # Adjust subplot params to make room
    g.figure.subplots_adjust(bottom=0.35)
    # g._legend.set_bbox_to_anchor((1, 0.5))  # Place legend outside
    # Draw a new legend with fewer columns
    handles = g._legend.legendHandles
    labels = [t.get_text() for t in g._legend.get_texts()]
    title = g._legend.get_title().get_text()

    g._legend.remove()  # Remove the original legend
    g.fig.legend(
        handles, labels, title=title, loc="center right", ncol=1, fontsize="10"
    )  # Add a new legend with desired settings
    g._legend.set_bbox_to_anchor((1, 0.5))

    # get rid of x axis label
    g.set(xlabel=None)

    # plt.tight_layout()
    plt.savefig(
        f"/data/tir/projects/tir5/users/mengyan3/rule_induction/src/figures/translation_metrics_colours_all.png"
    )
    plt.savefig(
        f"/data/tir/projects/tir5/users/mengyan3/rule_induction/src/figures/translation_metrics_colours_all.pdf"
    )

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        f"/data/tir/projects/tir5/users/mengyan3/rule_induction/src/figures/translation_metrics_colours_per_word_all.csv",
        index=False,
    )
    plt.gcf().clear()
