import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set a larger font scale and white background for better visibility
sns.set_theme(
    font_scale=1.5, style="whitegrid"
)  # Adjusted font_scale for overall increase

color_palette_methods = {
    "few-shot": "#DBB36B",
    "zs-cot": "#77DB69",
    "true-instruction": "#DB69B6",
    "instruction_induction:external_val": "#699BDB",
    "instruction_induction:verbal_confidence": "#699FDB",
    "instruction_induction:p(data)": "#699DBD",
    "instruction_induction:p(answer)": "#69DBDB",
}

old_to_new_names = {
    "base": "few-shot",
    "zs-cot": "zs-cot",
    "full_grammar": "true-instruction",
    "grammar_induction:ground_truth": "instruction_induction:external_val",
    "grammar_induction:p_data_given_hyp_guess": "instruction_induction:verbal_confidence",
    "grammar_induction:p_data_given_hyp_logprobs": "instruction_induction:p(data)",
    "grammar_induction:p_answer_given_hyp_logprobs": "instruction_induction:p(answer)",
}

df = pd.read_csv(
    "/data/tir/projects/tir5/users/mengyan3/rule_induction/mtob/baselines/scripts/compiled_results_mt.csv"
)
df["prompt_type"] = df["prompt_type"].replace(old_to_new_names)
df["model"] = df["model"].str.replace("meta-llama:Llama-2-70b-chat-hf", "Llama-2-70b")
df["model"] = df["model"].str.replace("meta-llama:Llama-2-7b-chat-hf", "Llama-2-7b")

ax = sns.catplot(
    data=df,
    x="model",
    y="chrf",
    hue="prompt_type",
    kind="bar",
    ci="sd",
    row="lang_direction",
    height=6,
    aspect=2,
    palette=color_palette_methods,
    legend=False,
)

# Optionally, move the legend outside of the plot for better visibility
ax.fig.subplots_adjust(top=0.9)  # Adjust subplot to make room

plt.legend(loc="upper right", title="Prompt Type", bbox_to_anchor=(1, 1.5), ncol=2)
# ax.set_xticklabels(
# ax.get_xticklabels(), rotation=45
# )  # Rotate x-axis labels for clarity

# Adjust bottom margin to make room for rotated x-axis labels
plt.subplots_adjust(bottom=0.15)
plt.tight_layout()
# Save figures with adjustments
plt.savefig("./src/figures/mtob_baselines.png")
plt.savefig("./src/figures/mtob_baselines.pdf")
