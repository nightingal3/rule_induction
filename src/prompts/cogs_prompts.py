PROBLEM_SOLVING_SYSPROMPT = "You are a linguist. Your job is to do semantic parsing of sentences, following the format outlined by the input/output pairs."
GRAMMAR_USING_SYSPROMPT = "You are a linguist. Your job is to do semantic parsing of sentences, following the grammar provided for you. Follow the format outlined by the input/output pairs."
GRAMMAR_INDUCTION_SYSPROMPT = "You are a linguist. Your job is to document the rules mapping sentences to their logical forms. Provide a grammar that captures the input/output examples provided."

base_prompt = {
    "system": PROBLEM_SOLVING_SYSPROMPT,
    "user": """Return the output preceded by 'Output:'\n{few_shot_examples}\nInput: {input}"""
}

prompt_for_grammar_induction = {
    "system": GRAMMAR_INDUCTION_SYSPROMPT,
    "user": """
    Write a grammar that captures the relationship between input phrases and outputs. Write the grammar in Backus-Naur form if possible.
    It's possible there are some more abstract rules that cannot be captured by Backus-Naur form, this is fine. They should also be recorded.
    {few_shot_examples}
    Grammar:
    """,
    "user_followup": """
    Use this grammar to parse the input example to get the correct output.

    {induced_grammar}

    However, just write the output like what's shown in these examples.
    {few_shot_examples}

    Return the output preceded by 'Output:'
    Input: {input}
    """
}

few_shot_examples_prompt = """Input: {input}\nOutput: {output}\n"""
