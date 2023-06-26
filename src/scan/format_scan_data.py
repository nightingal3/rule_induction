import datasets

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

    scan_simple_train.to_csv("./data/scan_simple_train.csv", index=False)
    scan_simple_test.to_csv("./data/scan_simple_test.csv", index=False)
    scan_length_train.to_csv("./data/scan_length_train.csv", index=False)
    scan_length_test.to_csv("./data/scan_length_test.csv", index=False)
    scan_addprim_jump_train.to_csv("./data/scan_addprim_jump_train.csv", index=False)
    scan_addprim_jump_test.to_csv("./data/scan_addprim_jump_test.csv", index=False)
