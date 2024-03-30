import pandas as pd
import pathlib
from collections import defaultdict, Counter
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pointbiserialr
import re

from src.utils.parsing_utils import parse_production_rule


ground_truth_dict = {
    "lug": "blue",
    "dax": "green",
    "wif": "red",
    "zup": "yellow",
    "bluf": "repeat twice",
    "walm": "repeat three times",
}


def errplot(x, y, yerr, hue, **kwargs):
    data = kwargs.pop("data")
    p = data.pivot_table(index=x, columns=hue, values=y, aggfunc="mean")
    err = data.pivot_table(index=x, columns=hue, values=yerr, aggfunc="mean")
    p.plot(kind="bar", yerr=err, ax=plt.gca(), **kwargs)


def merge_counters(counter1, counter2):
    for word, colors in counter2.items():
        for color, count in colors.items():
            counter1[word][color] += count
    return counter1


def check_against_dictionary(merged_counter, truth_dictionary):
    correctness_report = {}
    for word, correct_color in truth_dictionary.items():
        # Special handling for "bluf" and "walm" based on the presence of "repeat" in the rule
        if word in ["bluf", "walm"]:
            # Find all hypotheses for the word
            hypothesis_colours = merged_counter.get(word, {})
            correct_count = sum(
                count
                for color, count in hypothesis_colours.items()
                if "repeat" in color or "previous" in color
            )
            total_count = sum(hypothesis_colours.values())
            correctness = correct_count / total_count if total_count else 0
            correctness_report[word] = {
                "correctness_percentage": correctness * 100,
                "total_mentions": total_count,
            }
        else:
            hypothesis_colors = merged_counter.get(word, {})
            correct_count = hypothesis_colors.get(correct_color, 0)
            total_count = sum(hypothesis_colors.values())
            correctness = correct_count / total_count if total_count else 0
            correctness_report[word] = {
                "correctness_percentage": correctness * 100,
                "total_mentions": total_count,
            }
    return correctness_report


def evaluate_correctness_for_file(df_hyps, ground_truth_dict):
    """
    Evaluates the correctness of hypotheses within a single DataFrame (representing a file)
    and returns the correctness percentage for each word.
    """
    hypothesis_counter = defaultdict(lambda: defaultdict(int))
    for index, row in df_hyps.iterrows():
        for word, color in row["parsed_rules"].items():
            hypothesis_counter[word][color] += 1

    correctness_report = check_against_dictionary(hypothesis_counter, ground_truth_dict)
    return correctness_report


def evaluate_correctness_one_by_one(df_hyps, ground_truth_dict):
    """
    Evaluates the correctness of hypotheses within a single DataFrame (representing a file)
    and returns a dict of lists keyed by word where 0 indicates incorrect and 1 indicates correct.
    output: {word: (index, is_correct)}
    """
    correctness_report = defaultdict(list)
    for index, row in df_hyps.iterrows():
        task_id = row["task_id"]
        for word, color in row["parsed_rules"].items():
            if word in ["bluf", "walm"]:
                # Special handling for "bluf" and "walm" based on the presence of "repeat" in the rule
                if "repeat" in color or "previous" in color:
                    correctness_report[word].append((index, 1))
                else:
                    correctness_report[word].append((index, 0))
            else:
                if (
                    word not in ground_truth_dict
                ):  # making a hypothesis for a word that is not in the ground truth
                    correctness_report[word].append((index, 0))
                    continue
                if color == ground_truth_dict[word]:
                    correctness_report[word].append((index, 1))
                else:
                    correctness_report[word].append((index, 0))
    return correctness_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo")

    data = {}
    args = parser.parse_args()

    pattern = "**/*/*/*/*/*.csv"
    base_dir = pathlib.Path(
        "/data/tir/projects/tir5/users/mengyan3/rule_induction/logs"
    )
    predicted_word_dict = defaultdict(Counter)
    word_correctness = defaultdict(list)
    models = ["gpt-3.5-turbo", "gpt-4-turbo"]
    hyps_method = "p_answer_given_hyp_logprobs"

    model_names_for_final_report = []
    word_names_for_final_report = []
    mean_correctness_for_final_report = []
    corr_for_final_report = []
    p_val_for_final_report = []

    for model_name in models:
        base_df = None
        hyps_df = None
        for file_path in base_dir.glob(pattern):
            if not str(file_path).endswith("HYPSFILE.csv") and not "base" in str(
                file_path
            ):
                continue
            if "colours" not in str(file_path):
                continue
            if "miniscan" in str(file_path):
                continue
            if model_name not in str(file_path):
                continue

            if "base" in str(file_path):
                print("BASE: ", file_path)
                if base_df is None:
                    base_df = pd.read_csv(file_path)
                else:
                    base_df = pd.concat([base_df, pd.read_csv(file_path)])
            if "grammar_induction" in str(file_path) and hyps_method in str(file_path):
                print("HYPS: ", file_path)
                if hyps_df is None:
                    hyps_df = pd.read_csv(file_path)
                else:
                    hyps_df = pd.concat([hyps_df, pd.read_csv(file_path)])

        hyps_df["parsed_rules"] = hyps_df["all_hyps"].apply(
            lambda x: parse_production_rule(x)
        )
        hypothesis_counter_parsed = defaultdict(lambda: defaultdict(int))
        hypothesis_counter_unparsed = defaultdict(lambda: defaultdict(int))
        # Iterate through the DataFrame to update the counter
        correctness_report = evaluate_correctness_for_file(hyps_df, ground_truth_dict)

        df_first_per_vocab_and_task = (
            hyps_df.sort_values(by="all_probs", ascending=False)
            .groupby(["task_id", "for_word"])
            .first()
            .reset_index()
        )

        df_final_comparison = pd.merge(
            df_first_per_vocab_and_task, base_df, on="task_id", how="left"
        )

        correctness_report_one_by_one = evaluate_correctness_one_by_one(
            df_final_comparison, ground_truth_dict
        )

        # Flatten the correctness report into a single dictionary with index as key and correctness as value
        correctness_flat = {}
        for word, tuples in correctness_report_one_by_one.items():
            for index, is_correct in tuples:
                correctness_flat[index] = is_correct

        # Add a new column to df_final_comparison for correctness status
        # Use .apply(lambda x: correctness_flat.get(x.name, 0)) to map each row index to its correctness status
        df_final_comparison["hyp_correctness"] = df_final_comparison.apply(
            lambda x: correctness_flat.get(x.name, 0), axis=1
        )

        for word in ground_truth_dict:
            print("WORD: ", word)
            selected_rows = df_final_comparison[df_final_comparison["for_word"] == word]
            selected_rows["input_has_word"] = selected_rows["inputs"].apply(
                lambda x: (
                    word in re.search(r"Input: (.+?)Output:", x, re.DOTALL).group(1)
                    if re.search(r"Input: (.+?)Output:", x, re.DOTALL)
                    else False
                )
            )
            filtered_rows = selected_rows[selected_rows["input_has_word"] == True]
            corr, p_val = pointbiserialr(
                filtered_rows["correct"], filtered_rows["hyp_correctness"]
            )
            print("Hyp correctness on word: ", filtered_rows["hyp_correctness"].mean())
            print("Correlation: ", corr)
            print("p-value: ", p_val)

            model_names_for_final_report.append(model_name)
            word_names_for_final_report.append(word)
            mean_correctness_for_final_report.append(
                filtered_rows["hyp_correctness"].mean()
            )
            corr_for_final_report.append(corr)
            p_val_for_final_report.append(p_val)

        # Store correctness percentages for each word
        for word, report in correctness_report.items():
            word_correctness[word].append(report["correctness_percentage"])
        # for index, row in df_hyps.iterrows():
        #     for word, color in row["parsed_rules"].items():
        #         hypothesis_counter_parsed[word][color] += 1

        # for index, row in df_hyps.iterrows():
        #     hyp = row["all_hyps"]
        #     color = row["for_word"]
        #     hypothesis_counter_unparsed[color][hyp] += 1

        predicted_word_dict = merge_counters(
            predicted_word_dict, hypothesis_counter_parsed
        )

        # breakpoint()
        word_stats = {}
        for word, percentages in word_correctness.items():
            mean_correctness = np.mean(percentages)
            stderr_correctness = np.std(percentages, ddof=1) / np.sqrt(len(percentages))
            word_stats[word] = {
                "mean_correctness": mean_correctness,
                "stderr_correctness": stderr_correctness,
            }

        data[model_name] = word_stats

    final_report_df = pd.DataFrame(
        {
            "Model": model_names_for_final_report,
            "Word": word_names_for_final_report,
            "Mean Correctness": mean_correctness_for_final_report,
            "Correlation": corr_for_final_report,
            "p-value": p_val_for_final_report,
        }
    )
    print(final_report_df)
    final_report_df.to_csv("./annot/colours_correctness_report_panswer.csv")
    assert False
    df_list = []
    for model, words in data.items():
        for word, stats in words.items():
            df_list.append(
                {
                    "Model": model,
                    "Word": word,
                    "Mean Correctness": stats["mean_correctness"],
                    "Stderr Correctness": stats["stderr_correctness"],
                }
            )
    df = pd.DataFrame(df_list)

    # Plotting
    plt.figure(figsize=(14, 8))
    # sns.barplot(
    #     x="Word",
    #     y="Mean Correctness",
    #     hue="Model",
    #     data=df,
    #     capsize=0.1,
    #     palette="viridis",
    # )
    # unique_words = df["Word"].unique()
    # unique_models = df["Model"].unique()
    # word_positions = dict(zip(unique_words, range(len(unique_words))))

    # # Now plot the error bars for each model
    # for model in unique_models:
    #     model_df = df[df["Model"] == model]
    #     # Map the word to its position for the x-axis
    #     x_positions = [word_positions[word] for word in model_df["Word"]]
    #     plt.errorbar(
    #         x=x_positions,  # Use numerical x positions
    #         y=model_df["Mean Correctness"].to_numpy(),
    #         yerr=model_df["Stderr Correctness"].to_numpy(),
    #         fmt="none",
    #         capsize=3,
    #         capthick=1.5,
    #         lw=1.5,
    #     )
    df["Word"] = pd.Categorical(
        df["Word"], categories=df["Word"].unique(), ordered=True
    )

    # Create a catplot with error bars
    catplot = sns.catplot(
        data=df,
        kind="bar",
        x="Word",
        y="Mean Correctness",
        hue="Model",
        palette="viridis",
        capsize=0.1,
        height=6,
        aspect=2,
        legend_out=False,
        errorbar="sd",
    )

    # Iterate over the axes to add error bars
    # for ax in catplot.axes.flat:
    #     for i, bar in enumerate(ax.patches):
    #         # Get the data position
    #         hue_offset = i % len(df["Model"].unique())
    #         x = bar.get_x() + bar.get_width() * hue_offset / len(df["Model"].unique())
    #         # Get the error for this bar
    #         yerr = df.iloc[i]["Stderr Correctness"]
    #         ax.errorbar(
    #             x + bar.get_width() / 2,
    #             bar.get_height(),
    #             yerr=yerr,
    #             fmt="none",
    #             c="black",
    #             capsize=3,
    #         )

    # Adjust the layout, title and axis labels as needed
    catplot.set_axis_labels("Word", "Mean Correctness")

    plt.ylabel("Mean Correctness (%)")
    plt.xlabel("Word")
    plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.savefig("mean_correctness_by_word.png", bbox_inches="tight")
    plt.savefig("mean_correctness_by_word.pdf", bbox_inches="tight")
