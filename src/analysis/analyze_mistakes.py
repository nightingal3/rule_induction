import pandas as pd
import argparse

from src.utils.utils import find_examples_with_pattern, find_examples_with_all_patterns

def extract_last_input(rules_string: str):
    lines = rules_string.split('\n')
    try:
        input_string = lines[-1].split("Input: ")[1] 
    except:
        # search back through the lines for a line that says "Input: "
        input_string = ""
        for line in lines[::-1]:
            if "Input: " in line:
                input_string = line.split("Input: ")[1]
                break
        
    return input_string

def length_of_input_question(input_question: str):
    return len(input_question.split())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["colours", "scan"], required=True)
    parser.add_argument("--model", choices=["gpt-4", "gpt-3.5-turbo", "alpaca"], required=True)
    words = ["lug", "dax", "wif", "zup", "bluf", "walm"]
    args = parser.parse_args()
    
    df_base = pd.read_csv(f"./logs/{args.dataset}_simple_base_{args.model}_None_None_minset_True.csv")
    df_full_grammar = pd.read_csv(f"./logs/{args.dataset}_simple_full_grammar_{args.model}_None_None_minset_True.csv")
    df_rule_selection = pd.read_csv(f"./logs/{args.dataset}_simple_rule_selection_{args.model}_None_None_minset_False.csv")
    df_with_induced_grammar = pd.read_csv(f"./logs/{args.dataset}_simple_grammar_induction_{args.model}_None_None_minset_True.csv")
    
    df_base["input_question"] = df_base["inputs"].apply(extract_last_input)
    df_base["input_len"] = df_base["input_question"].apply(length_of_input_question)
    df_full_grammar["input_question"] = df_full_grammar["inputs"].apply(extract_last_input)
    df_full_grammar["input_len"] = df_full_grammar["input_question"].apply(length_of_input_question)
    df_rule_selection["input_question"] = df_rule_selection["inputs"].apply(extract_last_input)
    df_rule_selection["input_len"] = df_rule_selection["input_question"].apply(length_of_input_question)
    df_with_induced_grammar["input_question"] = df_with_induced_grammar["inputs"].apply(extract_last_input)
    df_with_induced_grammar["input_len"] = df_with_induced_grammar["input_question"].apply(length_of_input_question)

    print("=== Basic accuracy ===")
    print(f"Base: {df_base['correct'].mean()}")
    print(f"Full grammar: {df_full_grammar['correct'].mean()}")
    print(f"Rule selection: {df_rule_selection['correct'].mean()}")
    print(f"Grammar induction: {df_with_induced_grammar['correct'].mean()}")

    print("=== Number of different preds made ===")
    print(f"Base and full grammar: {df_base[df_base['correct'] != df_full_grammar['correct']].shape[0]}")
    print(f"Base and rule selection: {df_base[df_base['correct'] != df_rule_selection['correct']].shape[0]}")
    print(f"Base and grammar induction: {df_base[df_base['correct'] != df_with_induced_grammar['correct']].shape[0]}")

    print("=== Length of mistakes vs. correct ===")
    print("Overall mean length", df_base['input_len'].mean())
    print(f"Base: {df_base[df_base['correct'] == False]['input_len'].mean()}")
    print(f"Full grammar: {df_full_grammar[df_full_grammar['correct'] == False]['input_len'].mean()}")
    print(f"Rule selection: {df_rule_selection[df_rule_selection['correct'] == False]['input_len'].mean()}")
    print(f"Grammar induction: {df_with_induced_grammar[df_with_induced_grammar['correct'] == False]['input_len'].mean()}")

    print("=== Correctness of each word ===")
    for word in words:
        print(word.upper())
        print(f"base: {df_base[df_base['input_question'].str.contains(word)]['correct'].mean()}")
        print(f"full grammar: {df_full_grammar[df_full_grammar['input_question'].str.contains(word)]['correct'].mean()}")
        print(f"rule selection: {df_rule_selection[df_rule_selection['input_question'].str.contains(word)]['correct'].mean()}")
        print(f"grammar induction: {df_with_induced_grammar[df_with_induced_grammar['input_question'].str.contains(word)]['correct'].mean()}")

    # save mistakes to separate file
    df_base[df_base['correct'] == False].to_csv(f"./logs/{args.dataset}_simple_base_full_grammar_mistakes_{args.model}_None_None_minset_True.csv", index=False)
    df_full_grammar[df_full_grammar['correct'] == False].to_csv(f"./logs/{args.dataset}_simple_full_grammar_mistakes_{args.model}_None_None_minset_True.csv", index=False)
    df_rule_selection[df_rule_selection['correct'] == False].to_csv(f"./logs/{args.dataset}_simple_rule_selection_mistakes_{args.model}_None_None_minset_False.csv", index=False)
    df_with_induced_grammar[df_with_induced_grammar['correct'] == False].to_csv(f"./logs/{args.dataset}_simple_grammar_induction_mistakes_{args.model}_None_None_minset_True.csv", index=False)

    print(f"Mistakes written to {args.dataset}_simple_*_mistakes_{args.model}_None_None_minset_True.csv")
