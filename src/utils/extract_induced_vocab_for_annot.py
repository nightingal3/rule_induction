import pickle
import pandas as pd
import re
from typing import Tuple, List, Optional

def extract_vocab(model: str, category: str = "noun", answers_df: Optional[pd.DataFrame] = None) -> Tuple[List, List, List, List]:
    en_words = []
    chr_words = []
    categories = []
    model_names = []
    is_correct = []
    with open(f"./data/cherokee/en_chr_induced_{category}_{model}_shortest.pkl", "rb") as f:
        induced_vocab = pickle.load(f)
        en_w = list(induced_vocab.keys())
        en_clean_words = [re.sub(r'\s*\(.*\)', '', word).lower() for word in en_w]
        chr_w = list(induced_vocab.values())
        chr_clean_words = [re.sub(r'\s*\(.*\)', '', word) for word in chr_w]

        en_words.extend(en_clean_words)
        chr_words.extend(chr_clean_words)
        categories.extend([category] * len(induced_vocab))
        model_names.extend([model] * len(induced_vocab))

        if answers_df is not None:
            for c_w, e_w in zip(chr_clean_words, en_clean_words):
                true_word = answers_df[answers_df["en"] == e_w]["chr"].tolist()
                print(e_w, c_w, true_word)
                if len(true_word) == 0:
                    is_correct.append(None)
                else:
                    is_correct.append(true_word[0] == c_w)
                print(is_correct[-1])

    return en_words, chr_words, categories, model_names, is_correct

if __name__ == "__main__":
    en_words = []
    chr_words = []
    categories = []
    model_names = []
    correct = []
    
    answers = pd.read_csv("./data/cherokee/cherokee_common_wordlist.csv")

    for model in ["gpt-3.5-turbo", "gpt-4"]:
        for category in ["nouns", "verbs", "adjectives"]:
            en, chr, cat, mod, is_correct = extract_vocab(model, category, answers_df=answers)
            en_words.extend(en)
            chr_words.extend(chr)
            categories.extend(cat)
            model_names.extend(mod)
            correct.extend(is_correct)

    df = pd.DataFrame(
        {
            "en": en_words,
            "chr": chr_words,
            "category": categories,
            "model": model_names,
            "is_correct": correct
        }
    )
    df.to_csv("./data/cherokee/en_chr_induced_vocab_shortest.csv", index=False)