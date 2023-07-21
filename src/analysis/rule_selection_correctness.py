from typing import List
import pandas as pd
import argparse

def check_rules(rules_string: str, master_rules: List):
    lines = rules_string.split('\n')
    rules = lines[1:-2] 
    input_string = lines[-1].split(": ")[1]  

    # for each rule, split it into left hand and right hand parts
    rules_dict = {}
    for rule in rules:
        left_hand, right_hand = rule.split(" -> ")
        rules_dict[left_hand] = right_hand

    # split the input_string into separate words
    input_words = input_string.split()

    # check if all the input_words are in the left hand side of the rules
    for word in input_words:
        if word not in rules_dict.keys():
            return False

    # check if any master_rules that could apply are missed
    for rule in master_rules:
        left_hand, _ = rule.split(" -> ")
        if left_hand in input_words and rule not in rules:
            return False

    return True


if __name__ == "__main__":
    #gpt4_rules = pd.read_csv("./logs/colours_simple_rule_selection_gpt-4_None_None_minset_False.csv")
    gpt4_rules = pd.read_csv("./gpt-3.5-selected-rules.csv")

    # test with given rules_string and master_rules
    rules_induced = gpt4_rules["inputs"]
    master_rules = ["lug -> blue", "dax -> green", "wif -> red",
                    "zup -> yellow", "bluf -> repeat the last action twice", "walm -> repeat the last action three times"]

    num_correct = 0
    induced_rules_correct = []
    for rules_string in rules_induced:
        is_correct = check_rules(rules_string, master_rules)
        num_correct += int(is_correct)
        induced_rules_correct.append(is_correct)
    
    gpt4_rules["correct"] = induced_rules_correct
    #gpt4_rules.to_csv("./logs/colours_simple_rule_selection_gpt-4_None_None_minset_False.csv", index=False)
    gpt4_rules.to_csv("./gpt-3.5-selected-rules.csv")
    print(f"Accuracy of rules: {num_correct / len(rules_induced)}")







