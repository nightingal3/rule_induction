import pandas as pd
import os
import argparse
import re

# assume non-overlapping ranges since I launch jobs that way
def combine_into_df(logs_dir: str, dataset: str, model: str, prompt_style: str, split: str, minset: bool) -> pd.DataFrame:
    data = []
    for file_name in os.listdir(logs_dir):
        if file_name.startswith(f"{dataset}_{split}_{prompt_style}_{model}"):
            if minset and "minset_False" in file_name or not minset and "minset_True" in file_name:
                continue
            df = pd.read_csv(os.path.join(logs_dir, file_name))
            matches = re.search(r"_(\d+|None)_(\d+|None)", file_name)
            if matches:
                start_ind = matches.group(1) if matches.group(1) != "None" else 0
                end_ind = matches.group(2)
                df["start_ind"] = start_ind
                df["end_ind"] = end_ind
            data.append(df)
    # sort by start ind
    concat = pd.concat(data)
    concat["start_ind"] = concat["start_ind"].astype(int)
    concat = concat.sort_values(by=["start_ind"])
    return concat

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["gpt-3.5-turbo", "gpt-4", "alpaca", "llama-65b"], required=True)
    parser.add_argument("--dataset", type=str, choices=["scan", "cogs", "colours"], required=True)
    parser.add_argument("--prompt_style", type=str, choices=["base", "full_grammar", "grammar_induction"], required=True)
    parser.add_argument("--split", type=str, choices=["simple", "cp_recursion"], default="simple")
    parser.add_argument("--logs_dir", type=str, default="logs")
    parser.add_argument("--minset", type=bool, default=False, help="Whether a 'min cover' was used for in-context examples.")
    args = parser.parse_args()

    combined_df = combine_into_df(args.logs_dir, args.dataset, args.model, args.prompt_style, args.split, args.minset)

    combined_df.to_csv(f"./logs/consolidated_logs/{args.dataset}_{args.split}_{args.prompt_style}_{args.model}.csv", index=False)
    