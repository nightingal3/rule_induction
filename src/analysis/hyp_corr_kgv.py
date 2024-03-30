import pandas as pd
import json
from scipy.stats import pointbiserialr
import evaluate

chrf = evaluate.load("chrf")

model = "gpt-4-1106-preview"

hyps_path = f"/data/tir/projects/tir5/users/mengyan3/rule_induction/annot/finished/{model}_ek_annot.csv"
hyps_df = pd.read_csv(hyps_path)

base_path = f"/data/tir/projects/tir5/users/mengyan3/rule_induction/mtob/baselines/logs/kalamang/ek/{model}/base/temp_0.05/0/results_test_openai_{model}_temp_0.05_reference_sentences_2.json"
with open(base_path, "r") as f:
    base_data = json.load(f)

base_df = pd.DataFrame(base_data)
base_df["chrf_segment"] = base_df.apply(
    lambda row: chrf.compute(
        predictions=[row["text"]], references=[[row["ground_truth"]]]
    )["score"],
    axis=1,
)
base_df["task_id"] = base_df.index
base_df["base_is_correct"] = base_df.apply(
    lambda row: row["text"].lower() == row["ground_truth"].lower(), axis=1
)
breakpoint()
linked_df = pd.merge(hyps_df, base_df, on="task_id", how="left")
linked_df = linked_df.loc[linked_df["correct"] != "-1"]
corr, p_val = pointbiserialr(linked_df["correct"], linked_df["base_is_correct"])
print(f"Correlation: {corr}, p-value: {p_val}")
