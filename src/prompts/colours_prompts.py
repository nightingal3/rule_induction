GRAMMAR_INDUCTION_SYSPROMPT = "You are a rule induction system. Your job is to figure out the rules underlying a problem and report on them. Use the examples to guide your thinking."
PROBLEM_SOLVING_SYSPROMPT = "You are a problem solving system. Your job is to use the input-output pairs to solve the problem as well as you can."
GRAMMAR_USING_SYSPROMPT = "You are a parser. Carefully use the grammar to parse inputs to determine the correct output."

base_prompt = {
    "system": PROBLEM_SOLVING_SYSPROMPT,
    "user": """Return the output preceded by 'Output:'\n{few_shot_examples}\nInput: {input}"""
}

prompt_with_true_grammar = {
    "system": GRAMMAR_USING_SYSPROMPT,
    "user": """Use this grammar to parse the input example to get the correct output.

    Grammar:
    lug -> blue
    dax -> green
    wif -> red
    zup -> yellow
    bluf -> repeat the last action twice
    walm -> repeat the last action three times
    
    Examples:
    {few_shot_examples}

    Return the output preceded by 'Output:'
    Input: {input}
    """
}

prompt_for_grammar_induction = {
    "system": GRAMMAR_INDUCTION_SYSPROMPT,
    "user": """Write a grammar that captures the relationship between input phrases and outputs.
    It's possible there are some more abstract rules that cannot be captured by Backus-Naur form, this is fine. They should also be recorded.
    {few_shot_examples}
    Grammar:
    """,
    "user_followup": """Use this grammar to parse the input example to get the correct output.

    {induced_grammar}

    However, just write the output like what's shown in these examples.
    {few_shot_examples}

    Return the output preceded by 'Output:'
    Input: {input}
    """
}

few_shot_examples_prompt = """Input: {input}\nOutput: {output}\n"""

prompt_for_rule_selection = {
    "system": GRAMMAR_USING_SYSPROMPT,
    "user": """First, select rules from the grammar that are relevant to the current input. You should copy the rules that look relevant from the grammar below.
    Grammar:
    lug -> blue
    dax -> green
    wif -> red
    zup -> yellow
    bluf -> repeat the last action twice
    walm -> repeat the last action three times
    
    Input: {input}
    Rules:""",
    "user_followup": """Now, you should apply this subset of rules to the input to get the output:\n{rules}\n\nInput: {input}"""
}