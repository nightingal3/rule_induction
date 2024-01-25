from typing import Literal, Tuple, Optional
import random
import re
from functools import reduce

from src.prompts.functions_prompts import *
from src.task import BaseTask
from src.get_completion import get_completion_openai
from src.utils.utils import try_convert_to_float
from src.utils.parsing_utils import (
    parse_polynomial,
    get_ground_truth_answers,
    get_example_start_token,
    get_start_and_end_tokens,
)
from typing import Literal, List, Tuple


class FunctionsTask(BaseTask):
    task_type = "io"

    def __init__(
        self,
        prompt_style: Literal["base", "full_grammar", "grammar_induction", "zs-cot"],
        num_few_shot_examples: int = 5,
        seed: int = 42,
        degree: int = 1,
        dataset_size: int = 200,
        max_range: Tuple[int, int] = (-20, 20),
        num_questions_per_function: int = 5,
        **kwargs,
    ) -> None:
        assert (
            dataset_size >= num_questions_per_function
        ), "dataset_size must be >= num_questions_per_function"
        self.degree = degree
        assert (
            dataset_size % num_few_shot_examples == 0
        ), "dataset_size must be divisible by num_few_shot_examples"
        self.dataset_size = dataset_size // num_questions_per_function
        self.prompt_style = prompt_style
        self.seed = seed
        assert max_range[0] < max_range[1], "max_range must have first element < second"
        self.max_range = max_range
        self.num_few_shot_examples = num_few_shot_examples

        self.num_questions_per_function = num_questions_per_function
        random.seed(self.seed)

        self.rules = [
            self.sample_random_polynomial_coeffs() for _ in range(self.dataset_size)
        ]
        self.dataset, self.test_examples = self.make_random_dataset(dataset_size)
        self.proposed_hypotheses = {
            "hypothesis": [],
            "estimated_prob": [],
            "rank": [],
        }

    def __len__(self):
        return len(self.test_examples)

    def sample_random_polynomial_coeffs(self):
        coeffs = []
        for _ in range(self.degree + 1):
            coeff = random.randint(self.max_range[0], self.max_range[1])
            coeffs.append(coeff)
        return coeffs

    def make_input_output_pair(self, input: int, rule: List[int]):
        output = 0
        for i, coeff in enumerate(rule[::-1]):
            output += coeff * input**i
        return input, output

    def make_random_dataset(self, num_examples: int):
        dataset = []
        test_examples = []
        test_examples_to_rule_inds = []
        for i, rule in enumerate(self.rules):
            samples = []
            for _ in range(self.num_few_shot_examples):
                input = random.randint(self.max_range[0], self.max_range[1])
                samples.append(self.make_input_output_pair(input, rule))

            dataset.append(samples)

            curr_test_examples = []
            for _ in range(self.num_questions_per_function):
                test_input = random.randint(self.max_range[0], self.max_range[1])
                test_output = self.make_input_output_pair(test_input, rule)[1]
                curr_test_examples.append({"input": test_input, "output": test_output})
                test_examples_to_rule_inds.append(i)

            test_examples.extend(curr_test_examples)

        self.test_examples_to_rule_inds = test_examples_to_rule_inds
        return dataset, test_examples

    def _validate(self, idx: int, output_text: str):
        answer = self.test_examples[idx]["output"]
        try:
            # this was output: [1] before. Not sure why, I think sometimes the output is in a weird arbitrary place
            output_text = output_text.lower().split("output:")[0]
        except:
            try:
                output_text = int(output_text.strip())
            except:
                return False
        try:
            correct = answer == int(output_text.strip())
            return correct
        except:
            return False

    def validate(
        self, idx: int, output_text: str, expected_answer: Optional[int] = None
    ):
        if expected_answer is None:
            answer = self.test_examples[idx]["output"]
        else:
            answer = expected_answer

        # Normalize line breaks and lowercase
        output_text = output_text.replace("\r\n", "\n").lower()

        # Pattern to find "Output:" and the output number
        pattern = r"output:\s*(-?\d+)"
        matches = list(re.finditer(pattern, output_text, re.IGNORECASE | re.DOTALL))
        print("OUTPUT TEXT: ", output_text)
        print("MATCHES: ", [match.group() for match in matches])

        valid_output = None
        for match in matches:
            # Extract the context for this "Output:"
            start_pos = match.start()
            context = output_text[:start_pos]
            context_lines = context.split("\n")

            print("CONTEXT: ", context)
            # Check if the context does not contain "Input:" immediately before this "Output:"
            if len(context_lines) < 2 or "input:" not in context_lines[-2]:
                valid_output = match.group(1)

        if valid_output is not None:
            try:
                return answer == int(valid_output.strip())
            except ValueError:
                return False
        else:
            return False

    def get_input(self, idx: int):
        input = self.test_examples[idx]["input"]
        return input

    def get_answer(self, idx: int):
        output = self.test_examples[idx]["output"]
        return output

    def get_rule(self, idx: int):
        rules_idx = self.test_examples_to_rule_inds[idx]
        return self.rules[rules_idx]

    def get_few_shot_examples(self, idx: int):
        examples_idx = self.test_examples_to_rule_inds[idx]
        return self.dataset[examples_idx]

    def get_standard_prompt(self, idx: int) -> str:
        input = self.get_input(idx)
        few_shot_examples = self.get_few_shot_examples(idx)
        return self.standard_prompt_wrap(few_shot_examples, input), self.get_rule(idx)

    def get_full_grammar_prompt(
        self, idx: int, no_few_shot_examples: bool = False
    ) -> str:
        input = self.get_input(idx)
        few_shot_examples = self.get_few_shot_examples(idx)
        return self.full_grammar_prompt_wrap(
            few_shot_examples, input, idx, no_few_shot_examples=no_few_shot_examples
        )

    def get_zs_cot_prompt(self, idx: int) -> str:
        few_shot_examples = self.get_few_shot_examples(idx)
        few_shot_examples_str = "\n".join(
            [
                few_shot_examples_prompt.format(input=i, output=o)
                for i, o in few_shot_examples
            ]
        )
        return cot_zero_shot_prompt["user"].format(
            input=self.get_input(idx),
            few_shot_examples=few_shot_examples_str,
        )

    def get_special_prompt(
        self,
        idx: int,
        backend: str = "gpt-4",
        return_grammar_only: bool = False,
        no_few_shot_examples: bool = False,
        n_hyps: int = 1,
        rerank_by: str = "p_answer_given_hyp_logprobs",
        **kwargs,
    ) -> Tuple[
        str, int, int
    ]:  # TODO: this is really ugly and also the same everywhere. Refactor
        if self.prompt_style == "full_grammar":
            return (
                self.get_full_grammar_prompt(
                    idx, no_few_shot_examples=no_few_shot_examples
                ),
                0,
                0,
                self.get_rule(idx),
            )
        elif self.prompt_style == "zs-cot":
            return (
                self.get_zs_cot_prompt(idx),
                0,
                0,
                self.get_rule(idx),
            )
        else:
            if self.prompt_style == "grammar_induction":
                few_shot_examples = self.get_few_shot_examples(
                    idx
                )  # selecting some different examples for second step
                few_shot_examples_str = "\n".join(
                    [
                        few_shot_examples_prompt.format(input=i, output=o)
                        for i, o in few_shot_examples
                    ]
                )
                grammar_induction_prompt = self.get_grammar_induction_prompt(
                    idx, no_parse=True
                )
                if "gpt" in backend:
                    message = [
                        {"role": "system", "content": GRAMMAR_INDUCTION_SYSPROMPT},
                        {"role": "user", "content": grammar_induction_prompt},
                    ]

                    completion = self.get_best_grammar(
                        message,
                        backend,
                        n_hyps=n_hyps,
                        rerank_by=rerank_by,
                        few_shot_examples=few_shot_examples,
                    )
                    if isinstance(completion, list):
                        # returning all the possible hypotheses - save these

                        alternate_completions = [
                            c[0]["choices"][0]["message"]["content"] for c in completion
                        ]
                        alternate_probs = [c[1] for c in completion]
                        completion = completion[0][0]

                        self.proposed_hypotheses["hypothesis"].append(
                            alternate_completions
                        )
                        self.proposed_hypotheses["estimated_prob"].append(
                            alternate_probs
                        )
                        self.proposed_hypotheses["rank"].append(
                            list(range(len(alternate_completions)))
                        )

                    induced_grammar = completion["choices"][0]["message"]["content"]
                    # completion = get_completion_openai(message, backend, temp=0.0)

                    # induced_grammar = completion["choices"][0]["message"]["content"]

                    print(induced_grammar)
                    usage_completion = completion["usage"]["completion_tokens"]
                    usage_prompt = completion["usage"]["prompt_tokens"]

                else:  # alpaca or llama
                    raise NotImplementedError
                if return_grammar_only:
                    num_times_changed = 0
                    return (
                        induced_grammar,
                        usage_completion,
                        usage_prompt,
                        num_times_changed,
                    )

                prompt_with_induced_grammar = self.prompt_with_induced_grammar_wrap(
                    induced_grammar, few_shot_examples_str, self.get_input(idx)
                )
                return (
                    prompt_with_induced_grammar,
                    usage_completion,
                    usage_prompt,
                    self.get_rule(idx),
                )
            elif self.prompt_style == "rule_selection":
                rule_selection_prompt = self.get_rule_selection_prompt(idx)
                message = [
                    {"role": "system", "content": PROBLEM_SOLVING_SYSPROMPT},
                    {"role": "user", "content": rule_selection_prompt},
                ]
                completion = get_completion_openai(message, backend, temp=0.0)
                rules = completion["choices"][0]["message"]["content"]
                prompt_with_relevant_rules = self.prompt_with_relevant_rules_wrap(
                    rules, self.get_input(idx)
                )
                return (
                    prompt_with_relevant_rules,
                    completion["usage"]["completion_tokens"],
                    completion["usage"]["prompt_tokens"],
                )

    def get_best_grammar(
        self,
        message: list,
        backend: str,
        n_hyps: int,
        rerank_by: Literal[
            "p_data_given_hyp_guess",
            "p_answer_given_hyp_logprobs",
            "p_data_given_hyp_logprobs",
            "ground_truth",
        ],
        few_shot_examples: list,
        return_hyps_ranked: bool = True,
    ):
        few_shot_examples_str = "\n".join(
            [
                few_shot_examples_prompt.format(input=i, output=o)
                for i, o in few_shot_examples
            ]
        )
        assert n_hyps > 0, "n_hyps must be positive"
        if n_hyps == 1:
            # generate one hyp with temp = 0
            completion = get_completion_openai(message, backend, temp=1.0)
            induced_grammar = completion["choices"][0]["message"]["content"]
        else:
            # default temp 1.0 for different options
            # use turbo models' backends to get logprobs
            prob_model_names = {
                "gpt-4-0314": "gpt-4-1106-preview",
                "gpt-3.5-turbo-0613": "gpt-3.5-turbo-0613",
                "gpt-4-1106-preview": "gpt-4-1106-preview",
            }
            prob_model_name = prob_model_names[backend]
            completions = [
                get_completion_openai(message, prob_model_name, temp=1.0)
                for _ in range(n_hyps)
            ]
            if rerank_by == "p_data_given_hyp_guess":
                estimate_p_data_prompts = [
                    prompt_for_probability_guess.format(
                        few_shot_examples=few_shot_examples_str,
                        hypothesis=completion["choices"][0]["message"]["content"],
                    )
                    for completion in completions
                ]
                p_data_given_hyp_guesses = [
                    get_completion_openai(
                        [
                            {
                                "role": "system",
                                "content": "You are a probability estimating system. Your job is to judge how probable data is given an explanation, and answer only with a number from 0 to 1 inclusive.",
                            },
                            {"role": "user", "content": prompt},
                        ],
                        backend,
                        temp=0.0,
                        logprobs=True,
                    )["choices"][0]["message"]["content"]
                    for prompt in estimate_p_data_prompts
                ]
                p_data_given_hyp_guesses = [
                    try_convert_to_float(guess) for guess in p_data_given_hyp_guesses
                ]
                # rerank the hypotheses by probability
                completions = [
                    (completion, prob_guess)
                    for prob_guess, completion in sorted(
                        zip(p_data_given_hyp_guesses, completions),
                        key=lambda pair: pair[0],
                        reverse=True,
                    )
                ]

            elif rerank_by == "p_data_given_hyp_logprobs":
                # logprobs - from turbo models
                # oracle condition -> best possible function description in N generated hypotheses
                estimate_p_data_prompts = [
                    prompt_for_probability_logprobs.format(
                        few_shot_examples=few_shot_examples_str,
                        hypothesis=completion["choices"][0]["message"]["content"],
                    )
                    for completion in completions
                ]
                logprobs_estimations = [
                    get_completion_openai(
                        prompt, "davinci-002", temp=0.0, logprobs=True, max_tokens=0
                    )
                    for prompt in estimate_p_data_prompts
                ]
                # Exclude the hypothesis itself from the logprobs
                token_offsets = [
                    get_example_start_token(completion, start_indicator="Examples:")
                    for completion in logprobs_estimations
                ]
                sum_logprobs = []
                for i, (offset, completion) in enumerate(
                    zip(token_offsets, logprobs_estimations)
                ):
                    token_logprobs = completion["choices"][0]["logprobs"][
                        "token_logprobs"
                    ]

                    # Error checking
                    if offset >= len(token_logprobs):
                        print(
                            f"  Offset {offset} is out of range for token_logprobs. Skipping this completion."
                        )
                        sum_logprobs.append(-float("inf"))
                        continue

                    # Calculating sum of logprobs
                    sum_logprob = sum(token_logprobs[offset:]) / len(
                        token_logprobs[offset:]
                    )
                    sum_logprobs.append(sum_logprob)

                completions = [
                    (completion, sum_logprob)
                    for sum_logprob, completion in sorted(
                        zip(sum_logprobs, completions),
                        key=lambda pair: pair[0],
                        reverse=True,
                    )
                ]
            elif rerank_by == "p_answer_given_hyp_logprobs":
                estimate_p_data_prompts = [
                    prompt_for_probability_logprobs.format(
                        few_shot_examples=few_shot_examples_str,
                        hypothesis=completion["choices"][0]["message"]["content"],
                    )
                    for completion in completions
                ]
                logprobs_estimations = [
                    get_completion_openai(
                        prompt, "davinci-002", temp=0.0, logprobs=True, max_tokens=0
                    )
                    for prompt in estimate_p_data_prompts
                ]
                # Exclude the hypothesis itself from the logprobs
                token_offsets = [
                    get_example_start_token(completion, start_indicator="Examples:")
                    for completion in logprobs_estimations
                ]
                tokens_of_answers_only = []
                for i, (offset, completion) in enumerate(
                    zip(token_offsets, logprobs_estimations)
                ):
                    token_logprobs = completion["choices"][0]["logprobs"][
                        "token_logprobs"
                    ]
                    orig_text = completion["choices"][0]["text"]
                    char_offset = orig_text.find("Examples:") + len("Examples:")
                    orig_text = orig_text[char_offset:]

                    pattern_answer = "Output:\s*(-?\d+)"
                    answer_tok_ranges = get_start_and_end_tokens(
                        completion, pattern_answer
                    )
                    completion_logprobs_total = []
                    for start_token_ind, end_token_ind in answer_tok_ranges:
                        completion_logprobs_total.append(
                            sum(token_logprobs[start_token_ind:end_token_ind])
                        )

                    tokens_of_answers_only.append(completion_logprobs_total)

                sum_logprobs = [
                    sum(token_logprobs) / len(token_logprobs)
                    for token_logprobs in tokens_of_answers_only
                ]

                completions = [
                    (completion, sum_logprob)
                    for sum_logprob, completion in sorted(
                        zip(sum_logprobs, completions),
                        key=lambda pair: pair[0],
                        reverse=True,
                    )
                ]
            elif rerank_by == "ground_truth":
                parsed_equations = [
                    parse_polynomial(completion["choices"][0]["message"]["content"])
                    for completion in completions
                ]

                icl_example_answers_and_mses = [
                    (i, get_ground_truth_answers(parsed_eq, few_shot_examples))
                    for i, parsed_eq in enumerate(parsed_equations)
                ]

                completions = [
                    (completion, -mse)
                    for completion, (i, (_, mse)) in sorted(
                        zip(completions, icl_example_answers_and_mses),
                        key=lambda pair: pair[1][1][1],
                        reverse=False,
                    )
                ]

            if return_hyps_ranked:
                return completions

            return completions[0][0]

    def get_system_prompt(
        self,
    ) -> str:  # this should be the same for all tasks, abstract it higher
        if self.prompt_style == "full_grammar":
            return prompt_with_true_grammar["system"]
        elif self.prompt_style == "grammar_induction":
            return prompt_for_grammar_induction["system"]

        return base_prompt["system"]

    def full_grammar_prompt_wrap(
        self,
        few_shot_examples: str,
        input: str,
        idx: int,
        no_few_shot_examples: bool = False,
    ) -> str:
        # format as a x ** n ..., where n is the degree of the polynomial
        x_deg_strs = [f"x^{i}" for i in range(self.degree + 1)]
        coeffs = self.get_rule(idx)
        coeffs_strs = [str(c) for c in coeffs]
        coeffs_strs.reverse()
        str_rule = " + ".join(
            [
                f"{c} * {x}"
                for c, x in zip(coeffs_strs[::-1], x_deg_strs[::-1])
                if c != "0"
            ]
        )
        if no_few_shot_examples:
            few_shot_examples_str = "No examples."
        else:
            few_shot_examples_str = "\n".join(
                [
                    few_shot_examples_prompt.format(input=i, output=o)
                    for i, o in few_shot_examples
                ]
            )
        return prompt_with_true_grammar["user"].format(
            few_shot_examples=few_shot_examples_str,
            input=input,
            function=str_rule,
        )

    def get_grammar_induction_prompt(
        self, idx: int, no_few_shot_examples: bool = False, **kwargs
    ) -> str:
        few_shot_examples_str = "\n".join(
            [
                few_shot_examples_prompt.format(input=i, output=o)
                for i, o in self.get_few_shot_examples(idx)
            ]
        )
        return self.grammar_induction_prompt_wrap(
            few_shot_examples_str,
        )

    @staticmethod
    def standard_prompt_wrap(few_shot_examples, input):
        few_shot_examples_str = "\n".join(
            [
                few_shot_examples_prompt.format(input=i, output=o)
                for i, o in few_shot_examples
            ]
        )
        return base_prompt["user"].format(
            few_shot_examples=few_shot_examples_str, input=input
        )

    @staticmethod
    def grammar_induction_prompt_wrap(few_shot_examples: str) -> str:
        return prompt_for_grammar_induction["user_new"].format(
            few_shot_examples=few_shot_examples
        )

    @staticmethod
    def prompt_with_induced_grammar_wrap(
        induced_grammar: str, few_shot_examples: str, input: str
    ) -> str:
        return prompt_for_grammar_induction["user_followup"].format(
            induced_grammar=induced_grammar,
            input=input,
            few_shot_examples=few_shot_examples,
        )

    @staticmethod
    def grammar_induction_prompt_wrap(few_shot_examples: str) -> str:
        return prompt_for_grammar_induction["user_new"].format(
            few_shot_examples=few_shot_examples
        )
