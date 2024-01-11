from typing import List, Literal, Tuple, Optional, Dict

import pandas as pd
import random
import json
import jsonlines
import numpy as np
import os
import evaluate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import stanza 
import string
from collections import defaultdict

from src.prompts.naclo_prompts import * 
from src.task import BaseTask
from src.prompt_openai import get_completion_openai
from src.utils.utils import edit_distance
from src.utils.sentence_store import SentenceStore

class NacloTask(BaseTask):

    task_type = "translation"
    
    def __init__(self, prompt_style: Literal["base", "grammar_induction"], grammar_induction_loop: bool = False, split: Literal["dev", "test"] = "dev", **kwargs) -> None:
        self.split = split
        self.dev_problems = []
        for filename in os.listdir("./data/naclo-puzzlingmachines/data_public_reference_data_dev"):
            if filename.endswith(".json"):
                with open(f"./data/naclo-puzzlingmachines/data_public_reference_data_dev/{filename}") as f:
                    dev_problem = json.load(f)
                    background_info = {
                        "source_language": dev_problem["source_language"],
                        "target_language": dev_problem["target_language"],
                        "meta": dev_problem["meta"],
                        "train": dev_problem["train"]
                    }
                    for test_problem in dev_problem["test"]:
                        test_problem_data = {
                            "test": {
                                "source_language_test": dev_problem["source_language"] if test_problem[2] == ">" else dev_problem["target_language"],
                                "target_language_test": dev_problem["target_language"] if test_problem[2] == ">" else dev_problem["source_language"],
                                "input_test": test_problem[0] if test_problem[2] == ">" else test_problem[1],
                                "reference_test": test_problem[1] if test_problem[2] == ">" else test_problem[0],
                            }
                        }
                        self.dev_problems.append({**background_info, **test_problem_data})
                        
        self.test_problems = []
        for filename in os.listdir("./data/naclo-puzzlingmachines/data_public_data_test"):
            if filename.endswith(".json"):
                with open(f"./data/naclo-puzzlingmachines/data_public_data_test/{filename}") as f:
                    test_problem = json.load(f)
                    background_info = { 
                        "source_language": test_problem["source_language"],
                        "target_language": test_problem["target_language"],
                        "meta": test_problem["meta"],
                        "train": test_problem["train"]
                    }
                    for curr_test_problem in test_problem["test"]:
                        test_problem_data = {
                            "test": {
                            "source_language_test": test_problem["source_language"] if curr_test_problem[2] == ">" else test_problem["target_language"],
                            "target_language_test": test_problem["target_language"] if curr_test_problem[2] == ">" else test_problem["source_language"],
                            "input_test": curr_test_problem[0] if curr_test_problem[2] == ">" else curr_test_problem[1],
                            "reference_test": curr_test_problem[1] if curr_test_problem[2] == ">" else curr_test_problem[0],
                            }
                        }
                        self.test_problems.append({**background_info, **test_problem_data})

        self.test_data = self.dev_problems if split == "dev" else self.test_problems

        self.prompt_style = prompt_style
        self.grammar_induction_loop = grammar_induction_loop

        self.eval_criteria = ["chrf", "meteor", "sacrebleu"]
        self.eval_fns = {
            metric: evaluate.load(metric) for metric in self.eval_criteria
        }

        self.induced_rules = defaultdict(list)
        self.tfidf = {i: None for i in range(len(self.test_data))}
        self.nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse')
        self.sentence_store = {i: SentenceStore() for i in range(len(self.test_data))}

    def __len__(self) -> int:
        return len(self.test_data)

    def tfidf_vectorize(self) -> None:
        # Tf-idf vectorize the training data
        vectorizer = TfidfVectorizer(stop_words="english")
        for i in range(len(self.test_data)):
            eng_ind = 0 if self.test_data[i]["source_language"] == "english" else 1
            src_ind = eng_ind if self.test_data[i]["test"]["source_language_test"] == "english" else 1 - eng_ind
            self.tfidf[i] = vectorizer.fit_transform([demonstration[src_ind] for demonstration in self.test_data[i]["train"]])
            # test
            test_doc = self.test_data[i]["test"]["input_test"]
            test_tfidf = vectorizer.transform([test_doc])

    def process_en_svo(self) -> None:
        # Process training data to find minimal pairs.
        # Find subject, verb, and object for each english sentence.
        for idx in range(len(self.test_data)):
            eng_ind = 0 if self.test_data[idx]["source_language"] == "english" else 1
            demonstrations = self.test_data[idx]["train"]
            for d in demonstrations:
                en_sent = d[eng_ind]
                other_sent = d[1 - eng_ind]
                en_lemma_sent = " ".join([word.lemma for word in self.nlp(en_sent).sentences[0].words])
                doc = self.nlp(en_sent)
                for sent in doc.sentences:
                    categories = set()
                    unique_nouns = set()
                    unique_verbs = set()
                    unique_adjs = set()
                    for word in sent.words:
                        print(word.text, word.upos, word.feats)
                        if word.upos == "NOUN":
                            if word.feats is not None:
                                if "Number=Plur" in word.feats:
                                    categories.add("plural")
                                elif "Number=Sing" in word.feats:
                                    categories.add("singular")
                            unique_nouns.add(word.lemma)
                        elif word.upos == "VERB":
                            if word.feats is not None:
                                if "Tense=Past" in word.feats:
                                    categories.add("past")
                                elif "Tense=Pres" in word.feats:
                                    categories.add("present")
                            unique_verbs.add(word.lemma)
                        elif word.upos == "ADJ":
                            unique_adjs.add(word.lemma)
                        
                        category = word.upos
                        categories.add(category)

                    self.sentence_store[idx].add_sentence((en_sent, en_lemma_sent, other_sent), categories, unique_nouns, unique_verbs, unique_adjs)

    def guess_word_order(self, idx: int) -> None:
        word_order_prompt = """Here's a translation puzzle. Try to figure out the word order for sentences in the non-English language.\nExamples:\n{demonstrations}\n\nWrite your answer like this: 'RULE: [[english subject]] [[english verb]] [[english object]] -> <non-english word order>'\n\n"""
        demonstrations = self.get_demonstrations(idx)
        demonstrations_str = ""
        for d in demonstrations:
            demonstrations_str += f"english: {d[0]}\nother language:\n{d[1]}\n"
        prompt = word_order_prompt.format(demonstrations=demonstrations_str)
        message = [
            {"role": "system", "content": "Here's a translation puzzle. Try to figure out the word order for sentences in the non-English language."},
            {"role": "user", "content": prompt},
        ]
        completion = get_completion_openai(message, "gpt-4", temp=0.0)
        word_order = completion["choices"][0]["message"]["content"]
        print(word_order)
        self.induced_rules[idx].append(word_order)

    def guess_grammar_rules(self, idx: int) -> None:
        grammar_prompt = """Here's a translation puzzle. Here are two sets of sentences representing the difference between {set1_name} and {set2_name} in another language. Your job is to deduce the rule that underlies this difference, or say "no difference" if you believe no difference exists.\nExamples:\n{set1_name}:\n{demonstrations1}\n\n{set2_name}:\n{demonstrations2}\n\nWrite your answer as production rules if possible, such as 'RULE: <english> -> <other language>'. If you are not totally sure, please write 'no difference'.\n\n"""
        contrast_sets = self.sentence_store[idx].get_contrast_sets()
        for contrast_set in contrast_sets:
            set1_name, set2_name = contrast_set
            set1, set2 = contrast_sets[contrast_set]
            set1_demonstrations = ""
            for d in set1:
                set1_demonstrations += f"english: {d[0]}\nother language:\n{d[2]}\n"
            set2_demonstrations = ""
            for d in set2:
                set2_demonstrations += f"english: {d[0]}\nother language:\n{d[2]}\n"
            prompt = grammar_prompt.format(set1_name=set1_name, set2_name=set2_name, demonstrations1=set1_demonstrations, demonstrations2=set2_demonstrations)
            message = [
                {"role": "system", "content": "Here's a translation puzzle. Here are two sets of sentences representing the difference between {set1_name} and {set2_name} in another language. Your job is to deduce the rule that underlies this difference, or say 'no difference' if you believe no difference exists."},
                {"role": "user", "content": prompt},
            ]
            print("PROMPT: ", prompt)
            completion = get_completion_openai(message, "gpt-4", temp=0.0)
            grammar_rule = completion["choices"][0]["message"]["content"]
            print(grammar_rule)
            if "no difference" not in grammar_rule.lower():
                self.induced_rules[idx].append(grammar_rule)

    
    def find_similar_words(self, idx: int, test_word: str, src_lang: str) -> List[str]:
        docs_with_most_similar_words = []
        for demonstration in self.test_data[idx]["train"]:
            src_ind = 0 if self.get_meta(idx)["source_language"] == src_lang else 1
            sent_no_punct = demonstration[src_ind].replace(".", "")
            words = sent_no_punct.split()
            # merge words into phrases IF the entire phrase appears in one or more of the demonstrations

            if self.get_meta(idx)["source_language"] == "english":
                words = self.merge_words(words, idx)
            
            sentence_min_distance = float("inf")
            for word in words:
                distance = edit_distance(word.lower(), test_word.lower())

                if distance < sentence_min_distance:
                    sentence_min_distance = distance
            
            docs_with_most_similar_words.append((sentence_min_distance, demonstration[src_ind], demonstration[1 - src_ind]))
        
        sorted_docs_with_most_similar_words = sorted(docs_with_most_similar_words, key=lambda x: x[0])
        if sorted_docs_with_most_similar_words[0][0] == 0:
            return [doc for doc in sorted_docs_with_most_similar_words if doc[0] == 0]
        
        return sorted_docs_with_most_similar_words

    def merge_words(self, words: List[str], idx: int) -> List[str]:
        # TODO: test this
        merged_words = []
        has_been_merged = [False] * len(words)
        i = 0
        while i < len(words):
            curr_word = words[i]
            if has_been_merged[i]:
                i += 1
                continue
            if i + 1 < len(words):
                next_word = words[i + 1]
                merged_word = f"{curr_word} {next_word}"
                if self.sentence_store[idx].contains_sentence(merged_word):
                    merged_words.append(merged_word)
                    i += 2
                    continue
            merged_words.append(curr_word)
            i += 1
        return merged_words
    
    def get_vocab_induction_prompt(self, idx: int, threshold: int = 4) -> List[str]:
        def generator() -> str:
            curr_test_example = self.get_input(idx)
            test_words = curr_test_example.split()
            if self.get_meta(idx)["source_language"] == "english":
                test_words = self.merge_words(test_words)
            test_src_lang = self.test_data[idx]["test"]["source_language_test"].strip()
            test_tgt_lang = self.test_data[idx]["test"]["target_language_test"].strip()
            test_index = 0
            demonstrations_str = ""

            while test_index < len(test_words):
                word = test_words[test_index]
                demonstrations_by_relevance = self.find_similar_words(idx, word, src_lang=test_src_lang)
                relevant_demonstrations = [(demonstration[1], demonstration[2]) for demonstration in demonstrations_by_relevance if demonstration[0] <= threshold]
                demonstrations_str = ""
                for demonstration in relevant_demonstrations:
                    demonstrations_str += f"{test_src_lang}: {demonstration[0]}\n{test_tgt_lang}: {demonstration[1]}"
                meta = self.get_meta(idx)["meta"]
                yield vocab_induction_selected_examples_prompt.format(src_lang=test_src_lang, tgt_lang=test_tgt_lang, input=word, demonstrations=demonstrations_str, word_order=ground_truth_lang_features[test_src_lang], meta=meta)
                test_index += 1
            
            test_index = 0 # reset generator for next example
        
        return generator()
    
    def get_demonstrations(self, idx: int) -> List[str]:
        return self.test_data[idx]["train"]
    
    def get_input(self, idx: int) -> str:
        return self.test_data[idx]["test"]["input_test"]
    
    def get_answer(self, idx: int) -> dict:
        return self.test_data[idx]["test"]["reference_test"]
    
    def get_meta(self, idx: int) -> dict:
        return {"source_language": self.test_data[idx]["source_language"].strip(), "target_language": self.test_data[idx]["target_language"].strip(), "meta": self.test_data[idx]["meta"]}
    
    def get_few_shot_examples(self, idx: int) -> str:
        demonstrations = self.get_demonstrations(idx)
        metadata = self.get_meta(idx)
        examples_prompt = ""
        for demonstration in demonstrations:
            examples_prompt += few_shot_examples_src_tgt_prompt.format(src_lang=metadata["source_language"], tgt_lang=metadata["target_language"], input=demonstration[0], output=demonstration[1])
            examples_prompt += "\n"

        return examples_prompt 
       
    def validate(self, idx: int, output_text: str, return_ref_and_hyp_only: bool = False) -> dict:
        answer = self.get_answer(idx).lower()
        # they didn't standardize this char
        answer = answer.replace("’", "'")
        tgt_lang = self.test_data[idx]["test"]["target_language_test"].strip()
        try:
            output_text = output_text.lower().split(f"{tgt_lang}: ")[1]
        except:
            try:
                output_text = output_text.lower().split("translation: ")[1]
            except:
                print("You may want to check this output: ", output_text)
                output_text = output_text.lower()

        if not return_ref_and_hyp_only:
            scores = {
                metric: self.eval_fns[metric].compute(predictions=[output_text], references=[answer]) for metric in self.eval_criteria if metric != "bertscore"
            } # TODO: outsource translation specific stuff to a higher level mixin for the future
            scores["exact_match"] = output_text == answer
            
        if return_ref_and_hyp_only:
            return None, answer, output_text
        
        return scores
       
    def validate_all(self, refs: List, hyps: List) -> Dict[str, List]:
        scores = {
            metric: self.eval_fns[metric].compute(predictions=hyps, references=refs) for metric in self.eval_criteria if metric != "bertscore"
        }
        return scores
    
    def get_standard_prompt(self, idx: int) -> Tuple[str, str]:
        input = self.get_input(idx)
        few_shot_examples = self.get_few_shot_examples(idx)
        meta = self.get_meta(idx)
        src_lang = self.test_data[idx]["test"]["source_language_test"]
        tgt_lang = self.test_data[idx]["test"]["target_language_test"]
        meta_str = meta["meta"]

        return self.standard_prompt_wrap(src_lang, tgt_lang, meta_str, few_shot_examples, input), f"{src_lang}-{tgt_lang}"
    
    def get_special_prompt(self, idx: int, backend: str = "gpt-4", return_grammar_only: bool = False, use_cached: bool = False) -> Tuple[str, int, int]:
        if self.prompt_style == "full_grammar":
            return self.get_full_grammar_prompt(idx), 0, 0
        else:
            if self.prompt_style == "grammar_induction":
                few_shot_examples = self.get_few_shot_examples(idx + 1) # selecting some different examples for second step
                if use_cached:
                    induced_grammar = self.cached_induced_grammars[0]["grammar"]
                    usage_completion = 0
                    usage_prompt = 0
                else:
                    grammar_induction_prompt = self.get_grammar_induction_prompt(idx, no_parse=True)
                    if "gpt" in backend:
                        message = [
                            {"role": "system", "content": GRAMMAR_INDUCTION_SYSPROMPT},
                            {"role": "user", "content": grammar_induction_prompt},
                        ]
                        completion = get_completion_openai(message, backend, temp=0.0)
                        induced_grammar = completion["choices"][0]["message"]["content"]

                        converged = False
                        num_times_changed = 0
                        while self.grammar_induction_loop and not converged:
                            repeat_prompt = self.get_repeat_prompt(induced_grammar, schedule="target_hard_words", num_examples_to_get=3)
                            message = [
                                {"role": "system", "content": GRAMMAR_INDUCTION_SYSPROMPT},
                                {"role": "user", "content": repeat_prompt}
                            ]
                            completion = get_completion_openai(message, backend, temp=1)
                            induced_grammar_new = completion["choices"][0]["message"]["content"]
                            if "no changes" in induced_grammar_new.lower():
                                converged = True
                            else:
                                induced_grammar = induced_grammar_new
                                num_times_changed += 1
                        
                        print("TIMES CHANGED: ", num_times_changed)
                        print(induced_grammar)
                        usage_completion = completion["usage"]["completion_tokens"]
                        usage_prompt = completion["usage"]["prompt_tokens"]

                    else: # alpaca or llama
                        raise NotImplementedError
                    if return_grammar_only:
                        return induced_grammar, usage_completion, usage_prompt
                    
                prompt_with_induced_grammar = self.prompt_with_induced_grammar_wrap(induced_grammar, few_shot_examples, self.get_input(idx))
                return prompt_with_induced_grammar, usage_completion, usage_prompt
            elif self.prompt_style == "rule_selection":
                rule_selection_prompt = self.get_rule_selection_prompt(idx)
                message = [
                        {"role": "system", "content": PROBLEM_SOLVING_SYSPROMPT},
                        {"role": "user", "content": rule_selection_prompt},
                    ]
                completion = get_completion_openai(message, backend, temp=0.0)
                rules = completion["choices"][0]["message"]["content"]
                prompt_with_relevant_rules = self.prompt_with_relevant_rules_wrap(rules, self.get_input(idx))
                return prompt_with_relevant_rules, completion["usage"]["completion_tokens"], completion["usage"]["prompt_tokens"]
            elif self.prompt_style == "vocab_induction":
                vocab_induction_subproblems = self.get_vocab_induction_prompt(idx)
                induced_vocab = []
                for vocab_prompt in vocab_induction_subproblems:
                    message = [
                        {"role": "system", "content": VOCAB_INDUCTION_SYSPROMPT},
                        {"role": "user", "content": vocab_prompt},
                    ] 

                    completion = get_completion_openai(message, backend, temp=0.0)
                    completion_text = completion["choices"][0]["message"]["content"]
                    induced_vocab.append(completion_text)
                
                demonstrations = self.get_demonstrations(idx)
                src_lang = self.test_data[idx]["test"]["source_language_test"]
                tgt_lang = self.test_data[idx]["test"]["target_language_test"]
                demonstrations_str = ""
                for d in demonstrations:
                    demonstrations_str += f"{src_lang}: {d[0]}\n{tgt_lang}: {d[1]}\n\n"
                meta = self.get_meta(idx)["meta"]
                prompt_with_induced_vocab = vocab_induction_final_prompt.format(rules="\n".join(induced_vocab), tgt_lang=self.test_data[idx]["test"]["target_language_test"], src_lang=self.test_data[idx]["test"]["source_language_test"], input=self.get_input(idx), demonstrations=demonstrations_str, meta=meta)
                return prompt_with_induced_vocab, completion["usage"]["completion_tokens"], completion["usage"]["prompt_tokens"], f"{src_lang}-{tgt_lang}"
            
    def get_full_grammar_prompt(self, idx: int) -> str:
        input = self.get_input(idx)
        few_shot_examples = self.get_few_shot_examples(idx)
        return self.full_grammar_prompt_wrap(few_shot_examples, input)
    
    def get_grammar_induction_prompt(self, idx: int, no_parse: bool = False) -> str:
        few_shot_examples = self.get_few_shot_examples(idx, no_parse=no_parse)
        return self.grammar_induction_prompt_wrap(few_shot_examples,)
    
    def get_rule_selection_prompt(self, idx: int) -> str:
        input = self.get_input(idx)
        return self.rule_induction_prompt_wrap(input)
    
    def get_system_prompt(self) -> str:
        if self.prompt_style == "full_grammar":
            return prompt_with_true_grammar["system"]
        elif self.prompt_style == "grammar_induction":
            return prompt_for_grammar_induction["system"]
            
        return base_prompt["system"]
    
    def get_repeat_prompt(self, induced_grammar: str, schedule: str = "random", num_examples_to_get: Optional[int] = None) -> str:
        raise NotImplementedError
    

    @staticmethod
    def standard_prompt_wrap(src_lang: str, tgt_lang: str, meta: str, few_shot_examples: str, input: str) -> str:
        return base_prompt["user"].format(few_shot_examples=few_shot_examples, input=input, src_lang=src_lang, tgt_lang=tgt_lang, meta=meta)
    
    @staticmethod
    def full_grammar_prompt_wrap(few_shot_examples: str, input: str) -> str:
        return prompt_with_true_grammar["user"].format(few_shot_examples=few_shot_examples, input=input)
    
    @staticmethod
    def grammar_induction_prompt_wrap(few_shot_examples: str) -> str:
        return prompt_for_grammar_induction["user"].format(few_shot_examples=few_shot_examples)
    
    @staticmethod
    def prompt_with_induced_grammar_wrap(induced_grammar: str, few_shot_examples: str, input: str) -> str:
        return prompt_for_grammar_induction["user_followup"].format(induced_grammar=induced_grammar, input=input, few_shot_examples=few_shot_examples)

    @staticmethod
    def rule_induction_prompt_wrap(input: str) -> str:
        return prompt_for_rule_selection["user"].format(input=input)
    
    @staticmethod 
    def prompt_with_relevant_rules_wrap(rules: str, input: str) -> str:
        return prompt_for_rule_selection["user_followup"].format(rules=rules, input=input)
    
    @staticmethod
    def few_shot_examples_wrap(few_shot_inputs: List, few_shot_outputs: List) -> str:
        examples = ""
        for inp, oup in zip(few_shot_inputs, few_shot_outputs):
            example = few_shot_examples_prompt.format(input=inp, output=oup)
            examples += example + "\n"
        return examples