import pickle
import pandas as pd

from prompt_openai import get_completion_openai
from src import get_task
from src.prompts.cherokee_prompts import *

if __name__ == "__main__":
    _task = get_task("cherokee")
    task = _task("./data/cherokee/cherokee_simple_train.csv", "./data/cherokee/cherokee_simple_dev.csv", "./data/cherokee/cherokee_simple_dev.csv", prompt_style="base")
    common_vocab = pd.read_csv("./data/cherokee/cherokee_common_wordlist.csv")
    en_common_vocab = common_vocab["en"].tolist()
    chr_common_vocab = common_vocab["chr"].tolist()
    pos_list = common_vocab["category"].tolist()

    pos_to_POS = {"nouns": "NOUN", "verbs": "VERB", "adjectives": "ADJ"}
    # test basic vocab
    overall_scores, overall_scores_no_mapping = [], []
    word_in_output, word_in_output_no_mapping = [], []

    for en_word, chr_word, pos in zip(en_common_vocab, chr_common_vocab, pos_list):
        pos_tag = pos_to_POS[pos]
        sents_with_word = task.get_examples_containing_vocab(en_word, pos_tag, get_all_list=True)
        print(en_word, chr_word, pos_tag, len(sents_with_word))
        scores_for_word = {"bleu": [], "rouge": [], "meteor": []}
        scores_for_word_no_mapping = {"bleu": [], "rouge": [], "meteor": []}
        word_in_output_count, word_in_output_no_mapping_count = 0, 0
        for i, (src_sent, tgt_sent) in enumerate(sents_with_word):
            if i > 100:
                continue

            mapping = f"{en_word} -> {chr_word}"
            message = [
                {"role": "system", "content": PROBLEM_SOLVING_SYSPROMPT},
                {"role": "user", "content": prompt_with_induced_vocab.format(vocab_mappings=mapping, input=src_sent)},
            ]
            message_no_mapping = [
                {"role": "system", "content": PROBLEM_SOLVING_SYSPROMPT},
                {"role": "user", "content": prompt_one_example_only.format(input=src_sent)},
            ]
            output = get_completion_openai(message, model_name="gpt-3.5-turbo", temp=0.0)
            output_no_mapping = get_completion_openai(message_no_mapping, model_name="gpt-3.5-turbo", temp=0.0)
            output_text = output["choices"][0]["message"]["content"]
            output_text_no_mapping = output_no_mapping["choices"][0]["message"]["content"]

            if "English:" in output_text:
                output_text = output_text.split("English:")[1].strip()
            if "English:" in output_text_no_mapping:
                output_text_no_mapping = output_text_no_mapping.split("English:")[1].strip()

            scores = task.validate_explicit(tgt_sent, output_text)
            scores["bleu"] = scores["bleu"]["bleu"]
            scores["rouge"] = scores["rouge"]["rougeL"]
            scores["meteor"] = scores["meteor"]["meteor"]

            scores_no_mapping = task.validate_explicit(tgt_sent, output_text_no_mapping)
            scores_no_mapping["bleu"] = scores_no_mapping["bleu"]["bleu"]
            scores_no_mapping["rouge"] = scores_no_mapping["rouge"]["rougeL"]
            scores_no_mapping["meteor"] = scores_no_mapping["meteor"]["meteor"]

            print("scores", scores)
            print("scores_no_mapping", scores_no_mapping)

            en_word_in_output = en_word in output_text
            en_word_in_no_mapping_output = en_word in output_text_no_mapping
            
            print("en_word_in_output", en_word_in_output)
            print("en_word_in_no_mapping_output", en_word_in_no_mapping_output)

            word_in_output_count += en_word_in_output
            word_in_output_no_mapping_count += en_word_in_no_mapping_output

            for metric in scores:
                scores_for_word[metric].append(scores[metric])
            for metric in scores_no_mapping:
                scores_for_word_no_mapping[metric].append(scores_no_mapping[metric])
        
        for metric in scores_for_word:
            scores_for_word[metric] = sum(scores_for_word[metric]) / len(scores_for_word[metric])
        for metric in scores_for_word_no_mapping:
            scores_for_word_no_mapping[metric] = sum(scores_for_word_no_mapping[metric]) / len(scores_for_word_no_mapping[metric])

        overall_scores.append(scores_for_word)
        overall_scores_no_mapping.append(scores_for_word_no_mapping)
        word_in_output.append(word_in_output_count/min(len(sents_with_word), 100))
        word_in_output_no_mapping.append(word_in_output_no_mapping_count/min(len(sents_with_word), 100))

    print("overall_scores", overall_scores)
    print("overall_scores_no_mapping", overall_scores_no_mapping)

    print("Number of cases where mapping > no mapping")
    print(sum([1 if overall_scores[i]["rouge"] > overall_scores_no_mapping[i]["rouge"] else 0 for i in range(len(overall_scores))]))
    
    # save results
    with open("./logs/pkl/cherokee_induced_vocab_results.pkl", "wb") as f:
        pickle.dump({"overall_scores": overall_scores, "overall_scores_no_mapping": overall_scores_no_mapping, "word_in_output": word_in_output, "word_in_output_no_mapping": word_in_output_no_mapping}, f)
    
