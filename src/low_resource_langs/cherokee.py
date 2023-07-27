import datasets
import pandas as pd

def count_num_words(text):
    return len(text.split())

def count_num_chars(text):
    return len(text)

if __name__ == "__main__":
    dataset = datasets.load_dataset("chr_en")
    cherokee_simple_train, cherokee_simple_dev, cherokee_simple_test = dataset["train"], dataset["dev"], dataset["test"]
    cherokee_ood_train, cherokee_ood_dev, cherokee_ood_test = dataset["train"], dataset["out_dev"], dataset["out_test"]

    cherokee_simple_train = pd.DataFrame(cherokee_simple_train["sentence_pair"])
    cherokee_simple_dev = pd.DataFrame(cherokee_simple_dev["sentence_pair"])
    cherokee_simple_test = pd.DataFrame(cherokee_simple_test["sentence_pair"])
    cherokee_ood_train = pd.DataFrame(cherokee_ood_train["sentence_pair"])
    cherokee_ood_dev = pd.DataFrame(cherokee_ood_dev["sentence_pair"])
    cherokee_ood_test = pd.DataFrame(cherokee_ood_test["sentence_pair"])
    # also create a dataset of easy sentences since some of these are quite hard
    cherokee_simple_train["num_words"] = cherokee_simple_train["en"].apply(count_num_words)
    cherokee_simple_dev["num_words"] = cherokee_simple_dev["en"].apply(count_num_words)
    cherokee_simple_test["num_words"] = cherokee_simple_test["en"].apply(count_num_words)

    cherokee_easy_train = cherokee_simple_train[cherokee_simple_train["num_words"] <= 10]
    cherokee_easy_dev = cherokee_simple_dev[cherokee_simple_dev["num_words"] <= 10]
    cherokee_easy_test = cherokee_simple_test[cherokee_simple_test["num_words"] <= 10]

    cherokee_simple_train.to_csv("./data/cherokee/cherokee_simple_train.csv", index=False)
    cherokee_simple_dev.to_csv("./data/cherokee/cherokee_simple_dev.csv", index=False)
    cherokee_simple_test.to_csv("./data/cherokee/cherokee_simple_test.csv", index=False)
    cherokee_ood_train.to_csv("./data/cherokee/cherokee_ood_train.csv", index=False)
    cherokee_ood_dev.to_csv("./data/cherokee/cherokee_ood_dev.csv", index=False)
    cherokee_ood_test.to_csv("./data/cherokee/cherokee_ood_test.csv", index=False)
    cherokee_easy_train.to_csv("./data/cherokee/cherokee_easy_train.csv", index=False)
    cherokee_easy_dev.to_csv("./data/cherokee/cherokee_easy_dev.csv", index=False)
    cherokee_easy_test.to_csv("./data/cherokee/cherokee_easy_test.csv", index=False)