import pandas as pd
from pprint import pprint
from collections import OrderedDict

from src.utils.utils import find_examples_with_pattern, find_examples_with_all_patterns

command_types = {
        "and": "and",
        "after": "after",
        "twice": "twice",
        "thrice": "thrice",
        "opposite": "opposite",
        "around": "around",
        "left": "left",
        "right": "right",
        "turn left": "turn left",
        "turn right": "turn right",
        "walk": "walk",
        "look": "look",
        "run": "run",
        "jump": "jump",
        "turn opposite left": "turn opposite left",
        "turn opposite right": "turn opposite right",
        "turn around left": "turn around left",
        "turn around right": "turn around right",
        "u left": "around left",
        "u right": "around right",
    }

if __name__ == "__main__":
    actions_to_check = list(command_types.keys())
    data_df = pd.read_csv("./data/scan/scan_simple_train.csv")
    data_df_test = pd.read_csv("./data/scan/scan_simple_test.csv")
    results_df = pd.read_csv("./logs/scan_simple_base_gpt-4_0_100.csv")

    action_accs = {}
    output_order = ["walk", "jump", "look", "run", "turn left", "turn right", "u left", "u right", "after", "and", "opposite", "around", "twice", "thrice"]
    for action in actions_to_check:
        test_examples_with_action = find_examples_with_pattern(data_df_test, action)
        inds_in_first_100 = test_examples_with_action[test_examples_with_action.index < 100].index

        accuracy_on_action = results_df.iloc[inds_in_first_100]["correct"].mean()
        action_accs[action] = accuracy_on_action

    action_accs = OrderedDict({key: action_accs[key] for key in output_order})
    pprint(action_accs)
