import pickle

from src import get_task
from src.prompts.cherokee_prompts import *
from src.curriculum.sampler import compute_difficulty_word_len, make_compute_difficulty_freq
from prompt_openai import get_completion_openai

def get_induced_vocab(key: str, example_selection: str = "base"):
    vocab, upos = key.split("_")
    examples_prompt = task.get_examples_containing_vocab(vocab, upos, example_selection=example_selection)
    message = [
        {"role": "system", "content": VOCAB_INDUCTION_SYSPROMPT},
        {"role": "user", "content": examples_prompt},
    ]
    output = get_completion_openai(message, model_name="gpt-4", temp=0.0)
    output_text = output["choices"][0]["message"]["content"]
    if "no mapping" in output_text.lower():
        return vocab, None
    else:
        try:
            en_word, chr_word = output_text.split("->")
        except:
            # sometimes it copies English word -> Cherokee word from the example
            if "English word -> Cherokee word" in output_text:
                en_word, chr_word = output_text.split("English word -> Cherokee word")[-1].split("->")
            else:
                # else try to find the vocab word and grab whatever's after the -> 
                print(f"Can't parse output_text automatically: {output_text}")
                print("Please enter the values for en_word and chr_word.")
                print(f"en word: {vocab}")
                en_word = input("Enter en_word: ")
                chr_word = input("Enter chr_word: ")
    
    return en_word.strip(), chr_word.strip()

if __name__ == "__main__":
    _task = get_task("cherokee")
    task = _task("./data/cherokee/cherokee_simple_train.csv", "./data/cherokee/cherokee_simple_dev.csv", "./data/cherokee/cherokee_simple_dev.csv", prompt_style="base")
    #task.pos_tag_en_sents()
    
    #difficulty_by_len = compute_difficulty_word_len(list(task.train_data["en"]))
    #compute_difficulty_commonness = make_compute_difficulty_freq("./data/cherokee/most_common_pooled.pkl")
    #difficulty_by_commonness = compute_difficulty_commonness(list(task.en_tagged_sents))
    #combined_difficulty = [d1 + d2 for d1, d2 in zip(difficulty_by_len, difficulty_by_commonness)]
    # TODO: commonness needs to be normalized so that it's on the same scale as length

    en_common_nouns = pickle.load(open("./data/cherokee/most_common_nouns.pkl", "rb"))
    en_common_verbs = pickle.load(open("./data/cherokee/most_common_verbs.pkl", "rb"))
    en_common_adjectives = pickle.load(open("./data/cherokee/most_common_adjs.pkl", "rb"))
    en_common_function_words = pickle.load(open("./data/cherokee/most_common_cconjs.pkl", "rb"))

    most_common_nouns = {k: en_common_nouns[k] for k in list(en_common_nouns.keys())[:10]}
    most_common_verbs = {k: en_common_verbs[k] for k in list(en_common_verbs.keys())[:10]}
    most_common_adjectives = {k: en_common_adjectives[k] for k in list(en_common_adjectives.keys())[:10]}
    most_common_function_words = {k: en_common_function_words[k] for k in list(en_common_function_words.keys())[:5]}

    en_chr_function_words = {}
    for k, v in most_common_function_words.items():
        en_word, chr_word = get_induced_vocab(k, example_selection="shortest")
        print("Induced vocab:", en_word, chr_word)
        en_chr_function_words[en_word.strip()] = chr_word.strip()

    pickle.dump(en_chr_function_words, open("./data/cherokee/en_chr_induced_function_words_gpt-4_shortest.pkl", "wb"))
    assert False

    en_chr_nouns = {}
    for k, v in most_common_nouns.items():
        en_word, chr_word = get_induced_vocab(k, example_selection="shortest")
        print("Induced vocab:", en_word, chr_word)
        en_chr_nouns[en_word.strip()] = chr_word.strip()
    
    en_chr_verbs = {}
    for k, v in most_common_verbs.items():
        en_word, chr_word = get_induced_vocab(k, example_selection="shortest")
        print("Induced vocab:", en_word, chr_word)
        en_chr_verbs[en_word.strip()] = chr_word.strip()
    
    en_chr_adjectives = {}
    for k, v in most_common_adjectives.items():
        en_word, chr_word = get_induced_vocab(k, example_selection="shortest")
        print("Induced vocab:", en_word, chr_word)
        en_chr_adjectives[en_word.strip()] = chr_word.strip()

    # combine the dicts and save
    en_chr_induced_vocab = {**en_chr_nouns, **en_chr_verbs, **en_chr_adjectives}
    pickle.dump(en_chr_induced_vocab, open("./data/cherokee/en_chr_induced_vocab_gpt-4_shortest_pooled.pkl", "wb"))
    # save the induced vocab individually
    pickle.dump(en_chr_nouns, open("./data/cherokee/en_chr_induced_nouns_gpt-4_shortest.pkl", "wb"))
    pickle.dump(en_chr_verbs, open("./data/cherokee/en_chr_induced_verbs_gpt-4_shortest.pkl", "wb"))
    pickle.dump(en_chr_adjectives, open("./data/cherokee/en_chr_induced_adjectives_gpt-4_shortest.pkl", "wb"))

    
