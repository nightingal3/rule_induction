from typing import List, Literal, Tuple, Union, Dict, Optional

import pandas as pd
import random
import json
import jsonlines
import numpy as np
import os
import evaluate
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import stanza

from src.prompts.cherokee_prompts import * 
from src.task import BaseTask
from src.prompt_openai import get_completion_openai
from src.curriculum.sampler import CurriculumSamplerSimple
from src.utils import utils

class CherokeeTask(BaseTask):
    
    task_type = "translation"

    def __init__(self, train_file: str, test_file: str, dev_file: str, prompt_style: Literal["base", "full_grammar", "grammar_induction"], dictionary_file: Optional[str] = None, num_few_shot_examples: int = 5, nonce: bool = False, few_shot_min_set: bool = False, tgt_lang: Literal["chr", "en"] = "en", use_dev: bool = False, use_curriculum: bool = False, **kwargs) -> None:
        self.train_data, self.test_data, self.dev_data = pd.read_csv(train_file), pd.read_csv(test_file), pd.read_csv(dev_file)
        self.test_data = self.test_data if not use_dev else self.dev_data
        self.vocab_df = pd.read_csv(dictionary_file) if dictionary_file is not None else None

        self.tgt_lang = tgt_lang
        self.src_lang = "chr" if tgt_lang == "en" else "en"
        self.tgt_lang_name = "Cherokee" if tgt_lang == "chr" else "English"

        if self.vocab_df is not None:
            self.src_tgt_dict = dict(zip(self.vocab_df[self.src_lang], self.vocab_df[self.tgt_lang]))
            self.src_tgt_dict = {k: v for k, v in self.src_tgt_dict.items() if not pd.isna(v)} # empty translations

        self.base_prompt = base_prompt_chr_en if self.tgt_lang == "en" else base_prompt_en_chr
        self.few_shot_examples_prompt = few_shot_examples_chr_en_prompt if self.tgt_lang == "en" else few_shot_examples_en_chr_prompt

        self.prompt_style = prompt_style
        self.num_few_shot_examples = num_few_shot_examples
        self.is_nonce = nonce
        # if prompt style is base, this will just select the min set examples.
        # if prompt style is full_grammar, this will use the example parses as well.
        self.use_few_shot_minset = few_shot_min_set
        if self.use_few_shot_minset:
            self.min_examples = pd.read_csv("./data/colours/all_commands_minset.csv")

        self.cached_induced_grammars = []
        if os.path.exists("./data/colours/gpt_4_induced_grammars.jsonl"):
            with jsonlines.open("./data/colours/gpt_4_induced_grammars.jsonl", "r") as f:
                for line in f:
                    self.cached_induced_grammars.append(line)

        self.eval_criteria = ["sacrebleu", "chrf", "meteor"]
        self.eval_fns = {
            metric: evaluate.load(metric) for metric in self.eval_criteria
        }
        # add bertscore too.
        # bertscore can't do Cherokee, so we'll just do English.
        if self.tgt_lang == "en":
            self.eval_criteria.append("bertscore")
            self.eval_fns["bertscore"] = evaluate.load("bertscore", lang=self.tgt_lang)
        
        self.en_common_nouns = pickle.load(open("./data/cherokee/most_common_nouns.pkl", "rb"))
        self.en_common_verbs = pickle.load(open("./data/cherokee/most_common_verbs.pkl", "rb"))
        self.en_common_adjectives = pickle.load(open("./data/cherokee/most_common_adjs.pkl", "rb"))
        self.en_common_function_words = pickle.load(open("./data/cherokee/most_common_cconjs.pkl", "rb"))
        self.en_tagged_sents = []

    def __len__(self) -> int:
        return len(self.test_data)

    def pos_tag_en_sents(self) -> None:
        if len(self.en_tagged_sents) > 0:
            return

        sents = self.train_data["en"].tolist()
        nlp = stanza.Pipeline(lang="en", processors="tokenize,pos")
        for sent in sents:
            curr_sent = []
            tagged_sent = nlp(sent)
            for word in tagged_sent.sentences[0].words:
                curr_sent.append((word.text, word.upos))

            self.en_tagged_sents.append(curr_sent)

    def get_input(self, idx: int) -> str:
        return self.test_data.iloc[idx][self.src_lang]
    
    def get_answer(self, idx: int) -> str:
        return self.test_data.iloc[idx][self.tgt_lang]
    
    def get_few_shot_examples(self, idx: int, no_parse: bool = False) -> str:
        # get some random examples based on the idx (repeatable)
        if self.use_few_shot_minset:
            if self.prompt_style == "base" or no_parse:
                few_shot_inputs = self.min_examples[self.src_lang]
                few_shot_outputs = self.min_examples[self.tgt_lang]
                few_shot_formatted = self.few_shot_examples_wrap(few_shot_inputs, few_shot_outputs)
            else:
                all_examples = self.min_examples["example_parse_few_shot"]
                few_shot_formatted = "\n\n".join(all_examples)
        else:
            random.seed(idx)
            indices = random.sample(range(len(self.train_data)), self.num_few_shot_examples)
            few_shot_inputs = [self.train_data.iloc[i][self.src_lang] for i in indices]
            few_shot_outputs = [self.train_data.iloc[i][self.tgt_lang] for i in indices]
            few_shot_formatted = self.few_shot_examples_wrap(few_shot_inputs, few_shot_outputs)

        return few_shot_formatted
    
    def retrieve_few_shot_examples_by_similarity(self, idx: int, num_examples: int = 5, metric: str = "cos_sim") -> str:
        if self.tgt_lang == "chr" and metric == "cos_sim":
            raise ValueError("cos_sim is not supported for Cherokee, please use another metric.")
    
    def validate(self, idx: int, output_text: str, return_ref_and_hyp_only: bool = False) -> Dict[str, List]:
        if self.prompt_style == "base":
            output_actions = output_text.split(f"{self.tgt_lang_name}: ")[-1].strip()
        else:
            output_actions = output_text.split("Output: ")[-1].strip()

        # evaluate with MT metrics instead
        reference = self.test_data.iloc[idx][self.tgt_lang]

        if not return_ref_and_hyp_only:
            scores = {
                metric: self.eval_fns[metric].compute(predictions=[output_actions], references=[reference]) for metric in self.eval_criteria if metric != "bertscore"
            }
            if "bertscore" in self.eval_criteria:
                scores["bertscore"] = self.eval_fns["bertscore"].compute(predictions=[output_actions], references=[reference], lang=self.tgt_lang)
            
        # These are per-sentence BLEU scores. Use validate_all instead at the end to get the corpus level score.
        if return_ref_and_hyp_only:
            return None, reference, output_actions
        
        return scores
    
    def validate_all(self, refs: List, hyps: List) -> Dict[str, List]:
        scores = {
            metric: self.eval_fns[metric].compute(predictions=hyps, references=refs) for metric in self.eval_criteria if metric != "bertscore"
        }
        return scores
    
    def get_standard_prompt(self, idx: int) -> Tuple[str, str]: # TODO: refactor this signature later
        input = self.get_input(idx)
        few_shot_examples = self.get_few_shot_examples(idx)
        dict_prompt = None
        if self.src_tgt_dict is not None:
            matching_entries = utils.find_matching_dict_entries(input, self.src_tgt_dict)
            dict_prompt = self.dict_prompt_wrap(matching_entries)

        return self.standard_prompt_wrap(few_shot_examples, input, dict_entries=dict_prompt), ""
    
    def get_special_prompt(self, idx: int, backend: str = "gpt-3.5-turbo", return_grammar_only: bool = False, use_cached: bool = True) -> Tuple[str, int, int]:
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
                vocab_induction_prompt = self.get_vocab_induction_prompt(idx)
                message = [
                        {"role": "system", "content": PROBLEM_SOLVING_SYSPROMPT},
                        {"role": "user", "content": vocab_induction_prompt},
                    ]
                completion = get_completion_openai(message, backend, temp=0.0)
                vocab = completion["choices"][0]["message"]["content"]
                prompt_with_relevant_vocab = self.prompt_with_relevant_vocab_wrap(vocab, self.get_input(idx))
                return prompt_with_relevant_vocab, completion["usage"]["completion_tokens"], completion["usage"]["prompt_tokens"]

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
    
    def get_vocab_induction_prompts(self, input_sentence: str) -> List[str]:
        raise NotImplementedError
    
    def get_examples_containing_vocab(self, vocab: str, upos: str, get_all_list: bool = False, example_selection: Literal["base", "shortest", "informative"] = "base") -> Union[str, List[Tuple[str, str]]]:
        if upos == "NOUN":
            pos_data = self.en_common_nouns
        elif upos == "VERB":
            pos_data = self.en_common_verbs
        elif upos == "ADJ":
            pos_data = self.en_common_adjectives
        elif upos == "CCONJ":
            pos_data = self.en_common_function_words
        else:
            raise ValueError(f"POS {upos} not supported.")

        inds = list(pos_data[f"{vocab}_{upos}"]["inds"])
        if get_all_list:
            return [(self.train_data.iloc[ind][self.src_lang], self.train_data.iloc[ind][self.tgt_lang]) for ind in inds]
        
        if example_selection == "base":
            subset = self.train_data.iloc[inds].sort_values(by="num_words")[:10]
        elif example_selection == "shortest":
            subset = self.train_data.iloc[inds].sort_values(by="num_words")[:10]
            
        few_shot_examples = self.few_shot_examples_wrap(subset[self.src_lang].tolist(), subset[self.tgt_lang].tolist())
        prompt = prompt_for_vocab_induction["user"].format(few_shot_examples=few_shot_examples, word=vocab)

        return prompt
    
    def get_system_prompt(self) -> str:
        if self.prompt_style == "full_grammar":
            return prompt_with_true_grammar["system"]
        elif self.prompt_style == "grammar_induction":
            return prompt_for_grammar_induction["system"]
            
        return self.base_prompt["system"]
    
    def standard_prompt_wrap(self, few_shot_examples: str, input: str, dict_entries: Optional[str] = None) -> str:
        if dict_entries is None:
            return self.base_prompt["user"].format(few_shot_examples=few_shot_examples, input=input)
        else:
            return self.base_prompt["user_dict"].format(few_shot_examples=few_shot_examples, input=input, dict_entries=dict_entries)
    
    def full_grammar_prompt_wrap(self, few_shot_examples: str, input: str) -> str:
        return prompt_with_true_grammar["user"].format(few_shot_examples=few_shot_examples, input=input)
    
    def grammar_induction_prompt_wrap(self, few_shot_examples: str) -> str:
        return prompt_for_grammar_induction["user"].format(few_shot_examples=few_shot_examples)
    
    def prompt_with_induced_grammar_wrap(self, induced_grammar: str, few_shot_examples: str, input: str) -> str:
        return prompt_for_grammar_induction["user_followup"].format(induced_grammar=induced_grammar, input=input, few_shot_examples=few_shot_examples)

    def rule_induction_prompt_wrap(self, input: str) -> str:
        return prompt_for_rule_selection["user"].format(input=input)
    
    def prompt_with_relevant_rules_wrap(self, rules: str, input: str) -> str:
        return prompt_for_rule_selection["user_followup"].format(rules=rules, input=input)
    
    def dict_prompt_wrap(self, matching_entries: dict) -> str:
        matching_entries_str = "\n".join([f"{k} -> {v}" for k, v in matching_entries.items()])
        return prompt_for_dict_lookup.format(matching_entries=matching_entries_str)
    
    def few_shot_examples_wrap(self, few_shot_inputs: List, few_shot_outputs: List) -> str:
        examples = ""
        for inp, oup in zip(few_shot_inputs, few_shot_outputs):
            example = self.few_shot_examples_prompt.format(input=inp, output=oup)
            examples += example + "\n"
        return examples