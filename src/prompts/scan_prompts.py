
GRAMMAR_INDUCTION_SYSPROMPT = "You are a rule induction system. Your job is to figure out the rules underlying a problem and report on them. Use the examples to guide your thinking."
PROBLEM_SOLVING_SYSPROMPT = "You are a problem solving system. Your job is to use the input-output pairs to solve the problem as well as you can."
GRAMMAR_USING_SYSPROMPT = "You are a parser. Carefully use the grammar to parse inputs to determine the correct output."

base_prompt = {
    "system": PROBLEM_SOLVING_SYSPROMPT,
    "user": """Return the output preceded by 'Output:'\n{few_shot_examples}\nInput: {input}"""
}

prompt_with_true_grammar = {
    "system": GRAMMAR_USING_SYSPROMPT,
    "user": """
        Use this grammar to parse the input example to get the correct output.

        Grammar:
        C -> S and S
        C -> S after S
        C -> S
        S -> V twice
        S -> V thrice
        S -> V
        V -> D[1] opposite D[2]
        V -> D[1] around D[2] 
        V -> D
        V -> U
        D -> U left
        D -> U right
        D -> turn left
        D -> turn right
        U -> walk
        U -> look
        U -> run
        U -> jump

        Interpretation functions:
        [[walk]] = I_WALK
        [[look]] = I_LOOK
        [[run]] = I_RUN
        [[jump]] = I_JUMP
        [[turn left]] = I_TURN_LEFT
        [[turn right]] = I_TURN_RIGHT
        [[u left]] = I_TURN_LEFT [[u]]
        [[u right]] = I_TURN_RIGHT [[u]]
        [[turn opposite left]] = I_TURN_LEFT I_TURN_LEFT
        [[turn opposite right]] = I_TURN_RIGHT I_TURN_RIGHT
        [[u opposite left]] = [[turn opposite left]] [[u]]
        [[u opposite right]] = [[turn opposite right]] [[u]
        [[turn around left]] = I_TURN_LEFT I_TURN_LEFT I_TURN_LEFT I_TURN_LEFT
        [[turn around right]] = I_TURN_RIGHT I_TURN_RIGHT I_TURN_RIGHT I_TURN_RIGHT
        [[u around left]] = I_TURN_LEFT [[u]] I_TURN_LEFT [[u] I_TURN_LEFT [[u]] I_TURN_LEFT [[u] 
        [[u around right]] = I_TURN_RIGHT [[u]] I_TURN_RIGHT [[u] I_TURN_RIGHT [[u]] I_TURN_RIGHT [[u] 
        [[x twice]] = [[x]] [[x]]
        [[x thrice]] = [[x]] [[x]] [[x]]
        [[x and y]] = [[x]] [[y]]
        [[x after y]] = [[y]] [[x]]

        {few_shot_examples}

        Input: {input}
    """
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
    
    Input: {input}
    """
}

few_shot_examples_prompt = """Input: {input}\nOutput: {output}\n"""

