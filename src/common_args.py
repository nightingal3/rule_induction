import argparse


def get_common_arguments():
    parser = argparse.ArgumentParser(
        description="Common args shared across openai/open models"
    )
    parser.add_argument(
        "--temp", default=0.0, type=float, help="Temperature for sampling"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["scan", "cogs", "colours", "cherokee", "arc", "naclo", "functions"],
    )
    parser.add_argument(
        "--split",
        default="simple",
        choices=[
            "simple",
            "easy",
            "length",
            "jump",
            "cp_recursion",
            "prim_to_subj_common",
            "exposure_example_obj_proper",
            "obj_to_subj_common",
            "only_seen_as_unacc_subj_as_obj_omitted_transitive_subj",
            "debug",
            "AboveBelow",
            "CleanUp",
            "miniscan",
        ],
    )
    parser.add_argument(
        "--prompt_type",
        default="base",
        choices=[
            "base",
            "full_grammar",
            "grammar_induction",
            "rule_selection",
            "vocab_induction",
            "zs-cot",
        ],
    )
    parser.add_argument("--output", type=str)
    parser.add_argument("--start_ind", type=int)
    parser.add_argument("--end_ind", type=int)
    parser.add_argument("--num_few_shot_examples", type=int, default=5)
    parser.add_argument(
        "--use_min_cover",
        action="store_true",
        help="Use a curated set of few-shot examples that contain all primitives",
    )
    parser.add_argument("--return_induced_grammar_only", action="store_true")
    parser.add_argument(
        "--prompt_in_loop",
        help="Only for grammar induction. Present a few examples at a time until rules converge.",
        action="store_true",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Show debug info (prompts and num correct)"
    )
    parser.add_argument(
        "--no_few_shot_examples",
        action="store_true",
        help="Don't show few shot examples at all (for full_grammar and grammar_induction type prompts)",
    )
    parser.add_argument(
        "--rejection_sampling",
        action="store_true",
        help="Use rejection sampling to get a valid completion",
    )
    parser.add_argument(
        "--num_hyps",
        type=int,
        default=1,
        help="number of hypotheses to generate (for use with grammar_induction prompt type)",
    )
    parser.add_argument(
        "--hyp_reranking_method",
        type=str,
        choices=[
            "ground_truth",
            "p_data_given_hyp_guess",
            "p_data_given_hyp_logprobs",
            "p_answer_given_hyp_logprobs",
        ],
        default="ground_truth",
    )
    parser.add_argument(
        "--degree",
        help="polynomial degree for the functions domain",
        type=int,
        default=1,
    )

    return parser
