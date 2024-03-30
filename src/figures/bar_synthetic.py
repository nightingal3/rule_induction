import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# sns.set_theme(style="whitegrid", palette="Set2")
# set font size
sns.set_theme(font_scale=1.5, style="white", palette="Set2")
plot = "colours"  # "colours" or "functions"

color_palette_methods = {
    "few-shot": "#DBB36B",
    "zs-cot": "#77DB69",
    "true-instruction": "#DB69B6",
    "instruction_induction:external_val": "#699BDB",
    "instruction_induction:verbal_confidence": "#699FDB",
    "instruction_induction:p(data)": "#699DBD",
    "instruction_induction:p(answer)": "#69DBDB",
}
prompt_types_order = [
    "few-shot",
    "zs-cot",
    "true-instruction",
    "instruction_induction:external_val",
    "instruction_induction:verbal_confidence",
    "instruction_induction:p(data)",
    "instruction_induction:p(answer)",
]
old_to_new_names = {
    "base": "few-shot",
    "zs-cot": "zs-cot",
    "full_grammar": "true-instruction",
    "grammar_induction:ground_truth": "instruction_induction:external_val",
    "grammar_induction:p_data_given_hyp_guess": "instruction_induction:verbal_confidence",
    "grammar_induction:p_data_given_hyp_logprobs": "instruction_induction:p(data)",
    "grammar_induction:p_answer_given_hyp_logprobs": "instruction_induction:p(answer)",
}


if __name__ == "__main__":
    df = pd.read_csv(
        "/data/tir/projects/tir5/users/mengyan3/rule_induction/logs/compiled_results_aggregate_False.csv"
    )
    df_temp_0 = df[(df["temp"] == "temp_0.0") | (df["temp"] == "temp_0.1")]
    df_temp_0 = df_temp_0.loc[df_temp_0["model"] != "Llama-2-7b"]

    df_temp_1 = df[df["temp"] == "temp_1.0"]

    # df_colours = df_temp_0.loc[df_temp_0["domain"] == "colours"]
    # df_functions = df_temp_0.loc[df_temp_0["domain"] == "functions"]
    # df_functions = df_functions.loc[df_functions["prompt_type"] != "grammar_induction"]

    df_colours = df.loc[df["domain"] == "colours"]
    df_colours = df_colours.loc[df_colours["prompt_type"] != "grammar_induction"]
    df_functions = df.loc[df["domain"] == "functions"]
    df_functions = df_functions.loc[df_functions["prompt_type"] != "grammar_induction"]

    # map old to new names
    ()
    df_colours["prompt_type"] = df_colours["prompt_type"].replace(old_to_new_names)
    df_functions = df_functions.loc[
        df_functions["prompt_type"] != "p_answer_given_hyp_logprobs"
    ]
    df_functions["prompt_type"] = df_functions["prompt_type"].replace(old_to_new_names)

    df_selected = df_colours if plot == "colours" else df_functions
    # df_aggregated = (
    #     df_selected.groupby(["model", "prompt_type"])
    #     .agg(
    #         {"acc": "mean"}  # Replace "mean" with the appropriate aggregation function
    #     )
    #     .reset_index()
    # )

    # # Now create a complete DataFrame that includes all combinations of 'model' and 'prompt_type'
    # all_combinations = pd.MultiIndex.from_product(
    #     [df_aggregated["model"].unique(), prompt_types_order],
    #     names=["model", "prompt_type"],
    # )
    # df_complete = (
    #     df_aggregated.set_index(["model", "prompt_type"])
    #     .reindex(all_combinations)
    #     .reset_index()
    # )

    # Replace NaN values with a placeholder if needed
    # df_complete["acc"] = df_complete["acc"].fillna(None)

    # rename meta-llama:Llama-2-7b to Llama-2-7b
    df_selected["model"] = df_selected["model"].replace(
        "meta-llama:Llama-2-7b", "Llama-2-7b"
    )
    df_selected["model"] = df_selected["model"].replace(
        "meta-llama:Llama-2-70b", "Llama-2-70b"
    )

    models = df_functions["model"].unique()

    fig, axes = plt.subplots(
        1, len(models), figsize=(len(models) * 5, 4), sharey=False
    )  # Reduced height here
    num_bars = len(prompt_types_order)
    bar_positions = range(num_bars)

    for i, model in enumerate(models):
        ax = axes[i]
        df_model = df_selected[
            (df_selected["model"] == model) & df_selected["acc"].notna()
        ]

        model_prompt_types = df_model["prompt_type"].dropna().unique()
        # Get the bar positions for the current model
        bar_positions = range(len(model_prompt_types))

        for j, prompt_type in enumerate(model_prompt_types):
            mean_acc = df_model[df_model["prompt_type"] == prompt_type]["acc"].mean()
            se_acc = df_model[df_model["prompt_type"] == prompt_type]["acc"].sem()

            # Plot the bar at position j
            ax.bar(
                j,  # Bar position based on the actual prompt types present
                mean_acc,
                yerr=se_acc,
                color=color_palette_methods[prompt_type],
                label=prompt_type if i == 0 else "",  # Label only for the first subplot
                capsize=5,
            )
        # Customizing the y-axis scale per model if necessary
        # axes[i].set_ylim([your_min_value, your_max_value])

        # Set the title for each subplot to the model name
        if "meta-llama" in model:
            model = model.split(":")[1]
        axes[i].set_title(model)
        axes[i].set_xlabel("")
        axes[i].set_ylabel("")
        axes[i].set_xticklabels([])

    import matplotlib.patches as mpatches

    legend_handles = [
        mpatches.Patch(color=color, label=label)
        for label, color in color_palette_methods.items()
    ]
    plt.legend(
        handles=legend_handles,
        title="Prompt Type",
        loc="center left",
        bbox_to_anchor=(
            1,
            0.5,
        ),  # Adjust these values as needed to move the legend down
        ncol=1,
        shadow=True,
        fontsize=18,
        title_fontsize=20,
        fancybox=True,  # Optional, adds a rounded corner box with a lighter background
    )

    # Redraw the layout to take the new legend position into account
    fig.supylabel(
        "Accuracy (colour translation)", fontsize=14, va="center", ha="center"
    )

    plt.tight_layout()

    # g1.set_axis_labels("Model", "Accuracy on Linear Functions")

    plt.savefig(f"./src/figures/try_barplot_{plot}_T_all.png")
    plt.savefig(f"./src/figures/try_barplot_{plot}_T_all.pdf")
