import os
import pandas as pd
import glob
import pathlib

base_dir = pathlib.Path("/data/tir/projects/tir5/users/mengyan3/rule_induction/logs")

pattern = "**/*/*/*/*/*.csv"
aggregate = False

all_domains = []
all_models = []
all_prompt_types = []
all_temps = []
all_n_hyps = []
all_accs = []

for file_path in base_dir.glob(pattern):
    path = pathlib.Path(file_path)

    parts = path.parts

    start_ind = parts.index("logs")
    # if "p_data" in str(path):
    # breakpoint()
    parts = parts[start_ind + 1 :]
    domain = parts[0]
    model = parts[1]
    if "meta-llama" in model:
        model = model + ":" + parts[2]
        parts = parts[1:]

    prompt_type = parts[2]
    if len(parts) > 6:
        prompt_type = prompt_type + ":" + parts[3]
        temp = parts[4]
        n_hyps = parts[5]
        final_file = parts[6]
    else:
        temp = parts[3]
        n_hyps = 1
        final_file = parts[4]

    results_df = pd.read_csv(file_path)
    if "correct" not in results_df.columns:
        print("Error: no correct column in file", file_path)
        continue

    if len(results_df) < 200 and "miniscan" not in domain:
        print("Error: incomplete run", file_path)
        continue

    acc = results_df["correct"].mean()

    all_domains.append(domain)
    all_models.append(model)
    all_prompt_types.append(prompt_type)
    all_temps.append(temp)
    all_n_hyps.append(n_hyps)
    all_accs.append(acc)

results_df = pd.DataFrame(
    {
        "domain": all_domains,
        "model": all_models,
        "prompt_type": all_prompt_types,
        "temp": all_temps,
        "n_hyps": all_n_hyps,
        "acc": all_accs,
    }
)
results_df.sort_values(
    by=["domain", "model", "prompt_type", "temp", "n_hyps"], inplace=True
)

if aggregate:
    results_df = (
        results_df.groupby(["domain", "model", "prompt_type", "temp", "n_hyps"])["acc"]
        .agg(["var", "mean", "count"])
        .reset_index()
    )

    # Rename the columns for clarity
    results_df.rename(
        columns={"var": "variance", "mean": "mean_acc", "count": "num_experiments"},
        inplace=True,
    )

results_df.to_csv(f"logs/compiled_results_aggregate_{aggregate}.csv", index=False)
