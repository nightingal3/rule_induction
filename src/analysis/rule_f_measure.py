import csv
import argparse
import pandas as pd
from sklearn.metrics import f1_score
import re

def parse_rules(rules):
    parsed_rules = {}
    for rule in rules:
        # Remove any leading numbering and split only on '->'
        parts = rule.split('->')
        if len(parts) == 2:
            # Strip leading/trailing whitespace and take only the last word of the left-hand side
            key = parts[0].strip().split()[-1]
            value = parts[1].strip()
            parsed_rules[key] = value
    return parsed_rules

# Step 2: Load ground truth from a CSV file
def load_ground_truth(csv_file):
    ground_truth = {}
    with open(csv_file, mode='r') as infile:
        reader = csv.reader(infile)
        next(reader, None)  # skip the header if there is one
        for rows in reader:
            word, meaning, rule_type = rows
            ground_truth[word.strip()] = meaning.strip()
    return ground_truth

def compute_overall_micro_f1(ground_truth, prediction_sets):
    # Flatten the predictions and ground truths for micro F1 calculation
    all_true = []
    all_pred = []
    for nonce in ground_truth:
        all_true.extend([ground_truth[nonce]] * len(prediction_sets))
        all_pred.extend([prediction_set.get(nonce, "none") for prediction_set in prediction_sets])
    micro_f1 = f1_score(all_true, all_pred, average='micro', zero_division=0)
    return micro_f1

def compute_per_rule_f1(ground_truth, prediction_sets):
    # Compute per-rule F1 scores
    per_rule_f1_scores = {}
    for nonce in ground_truth:
        # Calculate the binary classification per nonce across all prediction sets
        true_labels = [1] * len(prediction_sets)  # The ground truth is always the rule
        predicted_labels = [(1 if prediction_set.get(nonce, None) == ground_truth[nonce] else 0) 
                            for prediction_set in prediction_sets]
        f1 = f1_score(true_labels, predicted_labels, zero_division=0)
        per_rule_f1_scores[nonce] = f1
    return per_rule_f1_scores
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground_truth", type=str, choices=["per-rule", "overall"], default="overall")
    # Replace 'ground_truth.csv' with the path to your CSV file
    ground_truth = load_ground_truth('/data/tir/projects/tir5/users/mengyan3/rule_induction/data/colours/colours.csv')
    df_model_rules = pd.read_csv("/data/tir/projects/tir5/users/mengyan3/rule_induction/logs/colours/colours_simple_grammar_induction_gpt-3.5-turbo_None_20_minset_True_loop_False_temp_1.0_few_shot_examples_False_induced_grammar.csv")
    all_model_rules = df_model_rules['outputs'].tolist()
    prediction_sets = [parse_rules(rules.split('\n')) for rules in all_model_rules]

    # Step 3: Compare predictions with ground truth
    # Compute per-rule F1
    per_rule_f1_scores = compute_per_rule_f1(ground_truth, prediction_sets)
    for nonce, f1_s in per_rule_f1_scores.items():
        print(f"Nonce {nonce}: F1 Score = {f1_s}")


    # Compute overall micro F1
    overall_micro_f1 = compute_overall_micro_f1(ground_truth, prediction_sets)
    print(f"Overall Micro F1 Score = {overall_micro_f1}")