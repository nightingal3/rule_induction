import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(font_scale=1.9, style="white")

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
)

legend = ax._legend
plt.setp(legend.get_texts(), fontsize="17")  # for legend text
plt.setp(legend.get_title(), fontsize="20")  # for legend title

# sns.move_legend(
#     ax,
#     "lower center",
#     bbox_to_anchor=(0.5, 1),
#     ncol=3,
#     title=None,
#     frameon=False,
# )

# tilt the x-axis labels
# plt.xticks(rotation=45)
# add more space below
# plt.subplots_adjust(bottom=0.35)

plt.savefig("./src/figures/mtob_baselines.png")
plt.savefig("./src/figures/mtob_baselines.pdf")
