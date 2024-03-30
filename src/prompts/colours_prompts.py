GRAMMAR_INDUCTION_SYSPROMPT = "You are a rule induction system. Your job is to figure out the rules underlying a problem and report on them. Use the examples to guide your thinking."
PROBLEM_SOLVING_SYSPROMPT = "You are a problem solving system. Your job is to use the input-output pairs to solve the problem as well as you can."
GRAMMAR_USING_SYSPROMPT = "You are a parser. Carefully use the grammar to parse inputs to determine the correct output."

llama_wrapper = """<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>

{user_prompt} [/INST]"""
base_prompt = {
    "system": PROBLEM_SOLVING_SYSPROMPT,
    "user": """Return the output preceded by 'Output:'\n{few_shot_examples}\nInput: {input}\nRemember to start your answer with "Output:"\n""",
}

cot_zero_shot_prompt = {
    "system": PROBLEM_SOLVING_SYSPROMPT,
    "user": """Return the output preceded by 'Final Output:'\n{few_shot_examples}\nLet's think step by step about what the translation could be. Work through your answer step by step and show your work. Remember to write down 'Final Output:' before your final answer.\nInput: {input}\n""",
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
    """,
}

prompt_with_true_grammar_miniscan = {
    "system": GRAMMAR_USING_SYSPROMPT,
    "user": """Use this grammar to parse the input example to get the correct output.

    Grammar:
    dax -> red
    lug -> blue
    wif -> green
    zup -> yellow
    fep -> repeat the last action twice
    [[x]] blicket [[y]] -> [[colour of x]] [[colour of y]] [[colour of x]] 
    [[X]] kiki [[Y]] -> [[translation of Y into colours]] [[translation of X into colours]]
    
    Examples:
    {few_shot_examples}

    Return the output preceded by 'Output:'
    Input: {input}""",
}

prompt_for_grammar_induction = {
    "system": GRAMMAR_INDUCTION_SYSPROMPT,
    "user_new": """Write a grammar that captures the relationship between input phrases and outputs.
    You should phrase this in terms of production rules. In other words, the format should be input phrase -> output phrase. Please write a rule for each input word that appears.
    You can also write rules that are more abstract like this: input phrase [[ something else ]] -> output phrase [[ something else different ]]. This captures contextual rules.
    {few_shot_examples}
    Grammar:""",
    "user_followup": """Use this grammar to parse the input example to get the correct output.

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

prompt_for_grammar_induction_one_word = """The below examples contain the nonce word {word}. Using the examples, deduce what {word} means.\n\n{few_shot_examples}\n\nWrite your answer like this: {word} -> meaning. Meaning can be a word or a general rule dependent on the context. Rule:\n"""

prompt_for_probability_guess = """These are examples of the translation of the word {word}.\n\n{few_shot_examples}?\n\nGiven these examples, how likely is this hypothesis about the meaning of {word}? {hypothesis}\n\nPlease give a probability between 0 and 1 inclusive, and only answer with a number.\nProbability:"""

all_induced_rules_wrapped = """Rules:\n{all_induced_rules}"""

prompt_for_probability_logprobs = """These are examples of applying this function: {hypothesis}:\nExamples:\n{few_shot_examples}"""
