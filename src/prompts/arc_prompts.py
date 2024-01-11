GRAMMAR_INDUCTION_SYSPROMPT = "You are a rule induction system. Your job is to figure out the rules underlying a problem and report on them. Use the examples to guide your thinking."
PROBLEM_SOLVING_SYSPROMPT = "You are a helpful assistant that solves analogy making puzzles. Only give the answer, no other words or text."
GRAMMAR_USING_SYSPROMPT = "You are a rule following system. Carefully use the induced rule to transform the input pattern to an output pattern. Only give the answer, no other words or text."

base_prompt = {
    "system": PROBLEM_SOLVING_SYSPROMPT,
    "user": """Let's try to complete the pattern:\n{few_shot_examples}\nInput: {input}\nOutput:"""
}

prompt_with_true_grammar = {
    "system": GRAMMAR_USING_SYSPROMPT,
    "user": """Here are some examples illustrating a rule:

    Examples:
    {few_shot_examples}

    Now here is the rule itself. Use this rule to transform the input grid to get the correct output grid.
    
    Rule:
    {induced_grammar}

    Return the output preceded by 'Output:'
    Input: {input}
    """,
    "1": "You will see a dashed or dotted horizontal line. You should copy everything above the line and the line itself, but fill in with 0s everything below the line."
}

prompt_for_grammar_induction = {
    "system": GRAMMAR_INDUCTION_SYSPROMPT,
    "user": """Write a rule that captures the relationship between the input grid and output grid.
    Try to be as general as possible and pay attention to shapes and lines.
    {few_shot_examples}
    Rule:
    """,
    "user_followup": """Here are some examples illustrating a rule:\nExamples:\n{few_shot_examples}\n\nNow here is the rule itself. Use this rule to transform the input grid to get the correct output grid.\nRule:\n\n{induced_grammar}\nReturn the output preceded by 'Output:'\nInput: {input}
    """,
    "user_invalid": """That may not be quite right. Here's a new example. Use this example to check whether or not your rule is correct.\nIf your rule is not correct, try generalizing your rule to also cover this case.\nIf everything is correct, write down "no changes".\n\n{new_example}\n\nKeep in mind that the rule you come up with should still be consistent with old examples:\n{few_shot_examples}\n\nNew rule:""",
}

few_shot_examples_prompt = """Input: {input}\nOutput: {output}\n"""