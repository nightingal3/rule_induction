import datasets
import pandas as pd
import stanza
from typing import List, Tuple, Optional
from collections import defaultdict
import pickle
from tqdm import tqdm

def count_num_words(text):
    return len(text.split())

def count_num_chars(text):
    return len(text)

def extract_en_features(nlp: stanza.Pipeline, sentences: List) -> Tuple[List, List]:
    features = defaultdict(lambda: {"count": 0, "feats": defaultdict(str), "inds": set()})

    print("Going through sentences...")
    for i, s in enumerate(tqdm(sentences)):
        doc = nlp(s)
        for sent in doc.sentences:
            for word in sent.words:
                word_txt = word.text.lower() + f"_{word.upos}"

                features[word_txt]["count"] += 1
                word_feats = []
                if word.feats is not None:
                    word_feats = word.feats.split("|")
                for feat in word_feats:
                    key, val = feat.split("=")
                    features[word_txt]["feats"][key] = val
                features[word_txt]["inds"].add(i)
    
    return features

def get_most_common(features: dict, upos_constraint: Optional[str] = None, feat_constraints: Optional[List[str]] = None) -> List[str]:
    if upos_constraint:
        features = {k: v for k, v in features.items() if k.endswith(upos_constraint)}
    if feat_constraints:
        features = {k: v for k, v in features.items() if all([v["feats"][c] for c in feat_constraints])}
    return dict(sorted(features.items(), key=lambda x: x[1]["count"], reverse=True))

if __name__ == "__main__":
    nlp = stanza.Pipeline(lang="en", processors="tokenize,pos")
    dataset = datasets.load_dataset("chr_en")
    cherokee_simple_train, cherokee_simple_dev, cherokee_simple_test = dataset["train"], dataset["dev"], dataset["test"]
    cherokee_ood_train, cherokee_ood_dev, cherokee_ood_test = dataset["train"], dataset["out_dev"], dataset["out_test"]

    cherokee_simple_train = pd.DataFrame(cherokee_simple_train["sentence_pair"])
    en_train_feats = extract_en_features(nlp, cherokee_simple_train["en"])
    most_common_nouns = get_most_common(en_train_feats, upos_constraint="NOUN")
    most_common_verbs = get_most_common(en_train_feats, upos_constraint="VERB")
    most_common_adjs = get_most_common(en_train_feats, upos_constraint="ADJ")
    most_common_function_words = get_most_common(en_train_feats, upos_constraint="CCONJ")

    with open("./data/cherokee/en_feats.pkl", "wb") as f:
        pickle.dump(en_train_feats, f)
    with open("./data/cherokee/most_common_nouns.pkl", "wb") as f:
        pickle.dump(most_common_nouns, f)
    with open("./data/cherokee/most_common_verbs.pkl", "wb") as f:
        pickle.dump(most_common_verbs, f)
    with open("./data/cherokee/most_common_adjs.pkl", "wb") as f:
        pickle.dump(most_common_adjs, f)
    with open("./data/cherokee/most_common_cconjs.pkl", "wb") as f:
        pickle.dump(most_common_function_words, f)

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