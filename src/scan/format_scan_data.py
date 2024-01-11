import datasets
import pandas as pd

def contains_pattern(text):
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
    matches = [key for key, pattern in command_types.items() if pattern in text]
    return matches if matches else None

if __name__ == "__main__":
    scan_simple = datasets.load_dataset("scan", "simple")
    scan_length = datasets.load_dataset("scan", "length")
    scan_addprim_jump = datasets.load_dataset("scan", "addprim_jump")
    scan_simple_train = scan_simple["train"]
    scan_simple_test = scan_simple["test"]
    scan_length_train = scan_length["train"]
    scan_length_test = scan_length["test"]
    scan_addprim_jump_train = scan_addprim_jump["train"]
    scan_addprim_jump_test = scan_addprim_jump["test"]

    scan_simple_train = scan_simple_train.to_pandas()
    scan_simple_train["patterns"] = scan_simple_train["commands"].apply(contains_pattern)
    scan_simple_test = scan_simple_test.to_pandas()
    scan_simple_test["patterns"] = scan_simple_test["commands"].apply(contains_pattern)
    scan_length_train = scan_length_train.to_pandas()
    scan_length_train["patterns"] = scan_length_train["commands"].apply(contains_pattern)
    scan_length_test = scan_length_test.to_pandas()
    scan_length_test["patterns"] = scan_length_test["commands"].apply(contains_pattern)
    scan_addprim_jump_train = scan_addprim_jump_train.to_pandas()
    scan_addprim_jump_train["patterns"] = scan_addprim_jump_train["commands"].apply(contains_pattern)
    scan_addprim_jump_test = scan_addprim_jump_test.to_pandas()
    scan_addprim_jump_test["patterns"] = scan_addprim_jump_test["commands"].apply(contains_pattern)
    
    scan_simple_train.to_csv("./data/scan_simple_train.csv", index=False)
    scan_simple_test.to_csv("./data/scan_simple_test.csv", index=False)
    scan_length_train.to_csv("./data/scan_length_train.csv", index=False)
    scan_length_test.to_csv("./data/scan_length_test.csv", index=False)
    scan_addprim_jump_train.to_csv("./data/scan_addprim_jump_train.csv", index=False)
    scan_addprim_jump_test.to_csv("./data/scan_addprim_jump_test.csv", index=False)
