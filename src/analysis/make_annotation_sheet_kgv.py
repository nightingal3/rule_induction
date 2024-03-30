import pandas as pd
import json
import evaluate

direction = "ek"
model = "gpt-4-1106-preview"
# json_data_path = f"/data/tir/projects/tir5/users/mengyan3/rule_induction/mtob/baselines/logs/kalamang/{direction}/{model}/grammar_induction/p_data_given_hyp_guess/temp_0.05/n_hyps_5/0/results_test_openai_{model}_temp_0.05_reference_sentences_2_reference_grammar_sketch_induced_5_p_data_given_hyp_guess.json"
json_data_path = "/data/tir/projects/tir5/users/mengyan3/rule_induction/mtob/baselines/logs/kalamang/ek/gpt-4-1106-preview/grammar_induction/p_data_given_hyp_guess/temp_0.05/n_hyps_5/0/results_test_openai_gpt-4-1106-preview_temp_0.05_reference_sentences_2_reference_grammar_sketch_induced_5_p_data_given_hyp_guess copy.json"
# csv_data_path = f"/data/tir/projects/tir5/users/mengyan3/rule_induction/mtob/baselines/logs/kalamang/{direction}/{model}/grammar_induction/p_data_given_hyp_guess/temp_0.05/n_hyps_5/0/results_test_openai_{model}_temp_0.05_reference_sentences_2_reference_grammar_sketch_induced_5_p_data_given_hyp_guess_p_data_given_hyp_guess_HYPSFILE.csv"
csv_data_path = "/data/tir/projects/tir5/users/mengyan3/rule_induction/mtob/baselines/logs/kalamang/ek/gpt-4-1106-preview/grammar_induction/p_data_given_hyp_guess/temp_0.05/n_hyps_5/0/results_test_openai_gpt-4-1106-preview_temp_0.05_reference_sentences_2_reference_grammar_sketch_induced_5_p_data_given_hyp_guess_p_data_given_hyp_guess_HYPSFILE.csv"
json_data = json.load(open(json_data_path))
# Load the JSON data into a DataFrame
df_json = pd.DataFrame(json_data)
# Create a new DataFrame with selected columns
df_json_selected = df_json[["ground_truth", "source", "text"]].copy()
df_json_selected.columns = ["tgt", "src", "hyp"]
# Add an index column to serve as task_id, assuming the index corresponds to task_id
df_json_selected["task_id"] = df_json_selected.index

# Load the CSV data into a DataFrame (replace the string loading with file loading in practice)
# For demonstration, loading from a string; replace with actual file loading in practice
chrf_metric = evaluate.load("chrf")
df_csv = pd.read_csv(csv_data_path, sep=",")

# Filter to keep only rank 0 hypotheses
df_csv_rank0 = df_csv[df_csv["rank"] == 0]

# Merge the DataFrames based on task_id
df_merged = pd.merge(df_csv_rank0, df_json_selected, on="task_id", how="left")
df_merged["chrf"] = df_merged.apply(
    lambda row: chrf_metric.compute(
        predictions=[row["hyp"]], references=[[row["tgt"]]]
    )["score"],
    axis=1,
)
df_merged[
    ["src", "tgt", "hyp", "hypothesis", "estimated_prob", "task_id", "chrf", "rank"]
]
# df_merged now contains linked data from both sources, with only rank 0 hypotheses
df_merged = df_merged[
    ["src", "tgt", "hyp", "hypothesis", "estimated_prob", "task_id", "chrf", "rank"]
]
df_merged.to_csv(f"./annot/{model}_{direction}_annot.csv")
print(f"Saved to ./annot/{model}_{direction}_annot.csv")
