GRAMMAR_INDUCTION_SYSPROMPT = "You are a pattern recognition system. Your job is to come up with a function that describes the data as well as you can."
PROBLEM_SOLVING_SYSPROMPT = "You are a problem solving system. Your job is to use the input-output pairs to solve the problem as well as you can."
GRAMMAR_USING_SYSPROMPT = "You are a problem solving system. Your job is to apply the function to the data in order to produce an answer."

llama_wrapper = """<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>

{user_prompt} [/INST]"""
base_prompt = {
    "system": PROBLEM_SOLVING_SYSPROMPT,
    "user": """Return the output preceded by 'Output:'\n{few_shot_examples}\nInput: {input}""",
}

cot_zero_shot_prompt = {
    "system": PROBLEM_SOLVING_SYSPROMPT,
    "user": """Return the output preceded by 'Output:'\n{few_shot_examples}\nInput: {input}\nLet's think step by step. Remember to write down 'Output:' before your final answer.""",
}

prompt_with_true_grammar = {
    "system": GRAMMAR_USING_SYSPROMPT,
    "user": """Use this function to apply to the input example to get the correct output.

    Function:
    {function}
    
    Examples:
    {few_shot_examples}

    Return the output preceded by 'Output:'
    Input: {input}
    """,
}

prompt_for_grammar_induction = {
    "system": GRAMMAR_INDUCTION_SYSPROMPT,
    "user_new": """Write the function that captures the relationship between inputs and outputs. You should write it in the form y = ax^0 + bx^1.
    {few_shot_examples}
    Function (please write explicitly in the exact form Output: y = ax^0 + bx^1):""",
    "user_followup": """Use this function to apply to the input example to get the correct output.

    {induced_grammar}

    However, just write the output like what's shown in these examples.
    {few_shot_examples}

    Return the output preceded by 'Output:'
    Input: {input}
    """,
    "user_repeat": """Here are some new examples. Use these examples to check whether or not your grammar is correct. If your grammar is not correct, please write down a revised version.
    If everything is correct, write down "no changes".
    
    {few_shot_examples}

    Your current grammar:
    {induced_grammar}

    New grammar:
    """,
}

few_shot_examples_prompt = """Input: {input}\nOutput: {output}\n"""
# TODO: write good prompts for the rest of this

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
    "user_followup": """Now, you should apply this subset of rules to the input to get the output:\n{rules}\n\nInput: {input}""",
}

prompt_for_probability_guess = """How likely is this hypothesis about the function to be true given the data?\n\nExamples: {few_shot_examples}\n\nFunction explanation: {hypothesis}\n\nPlease give a probability between 0 and 1 inclusive, and only answer with a number.\nProbability:"""
prompt_for_probability_logprobs = """These are examples of applying this function: {hypothesis}:\nExamples:\n{few_shot_examples}"""
