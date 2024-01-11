GRAMMAR_INDUCTION_SYSPROMPT = "You are a linguist. Your task is to deduce a grammar for Cherokee based on the examples given. The grammar will be used to translate between Cherokee and English."
PROBLEM_SOLVING_SYSPROMPT = "You are an expert translator. You are translating between Cherokee and English. Use the examples to guide your thinking."
GRAMMAR_USING_SYSPROMPT = "You are an expert translator. You are translating between Cherokee and English. You are provided with a grammar, which you should use to parse the examples and translate them."
VOCAB_INDUCTION_SYSPROMPT = "You are a linguist. Your task is to deduce a vocabulary for Cherokee based on the examples given. The vocabulary will be used to translate between Cherokee and English."
# grammar comes from this page: https://www.cherokeelessons.com/content/Cherokee-Language-Grammar---Cherokee-Messenger-1844-1846/
# nouns come from this word list: https://language.cherokee.org/word-list/
cherokee_grammar_components = {
    "pronouns": """Common pronouns:
    ᎠᏴ -> I
    ᎠᏴ -> we
    ᏂᎯ -> you
    Ꮎ -> that/he/she
    ᎾᏍᎩ -> that/he/she
    ᎯᏯ -> this
    ᎯᎠ -> this

    Reflexive pronouns:
    ᎠᏋᏒ -> myself
    ᏨᏒ -> yourself
    ᎤᏩᏒ -> himself
    ᎩᏅᏒ -> ourselves
    ᎢᎬᏒ -> ourselves
    ᎣᎬᏒ -> ourselves
    ᎣᎩᏅᏒ -> ourselves
    ᏍᏛᏒ -> yourselves (two)
    ᎢᏨᏒ -> yourselves (three or more)
    ᎤᏅᏒ -> themselves

    Singular possessive pronouns:
    ᎠᏆᏤᎵ -> mine
    ᏣᏤᎵ -> thine
    ᎤᏤᎵ -> his
    ᎩᎾᏤᎵ -> ours (mine and yours)
    ᎣᎩᎾᏤᎵ -> ours (his and mine)
    ᏍᏓᏤᎵ -> yours (two)
    ᎢᎦᏤᎵ -> ours (yours and mine)
    ᎣᎦᏤᎵ -> ours (theirs and mine)
    ᎢᏣᏤᎵ -> yours (three or more)
    ᎤᎾᏤᎵ -> theirs

    Pronouns about people:
    ᏥᎦᏙᎦ -> the one who is standing
    ᏤᏙᎠ -> the one who is moving about
    ᏧᏬᎳ -> the one who is sitting
    ᏥᎦᏅᎦ -> the one who is lying down
    ᏨᏓᏯᎢ -> the one who is coming
    ᏥᏩᎢ -> the one who is going
    ᏥᏲᎱᏒ -> the one who is dead
    ᏤᎭ -> the one who is living
    ᏧᏢᎦ -> the one who is sick
    """,
    "verb_conjugation": """Pronoun prefixes:
    Ꮵ[[verb]] -> I [[verb]]
    Ꭿ[[verb]] -> you [[verb]]
    Ꭷ[[verb]] -> he/she [[verb]]
    ᎢᏂ[[verb]] -> we (you and I) [[verb]]
    ᎣᏍᏗ[[verb]] -> we (him/her and I) [[verb]]
    ᏍᏗ[[verb]] -> you (two) [[verb]]
    ᎢᏗ[[verb]] -> we (you and I) [[verb]]
    ᎣᏥ[[verb]] -> we (they and I) [[verb]]
    ᎢᏥ[[verb]] -> you (three or more) [[verb]]
    ᎠᏂ[[verb]] -> they [[verb]]

    Tense suffixes:
    [[verb]]Ꭶ -> am [[verb]]ing
    [[verb]]ᎪᎢ -> am [[verb]]ing (habitually)
    [[verb]]ᎬᎩ -> was [[verb]]ing (with knowledge)
    [[verb]]ᎨᎢ -> was [[verb]]ing (without knowledge)
    [[verb]]ᎨᏍᏗ -> will be [[verb]]ing
    [[verb]]ᎬᎢ -> my [[verb]]ing
    """,
    "common_verbs": """Common verbs:
    ᏬᏂᎭ -> to speak
    ᏛᏁ -> to do
    """,
    "common_words": """Common words:
    ᎠᏏᏴᏫ -> person
    ᎠᏍᎦᏯ -> man
    ᎠᎨᏴ -> woman
    ᎠᏧᏣ -> boy
    ᎠᎨᏳᏣ -> girl
    ᎠᏕᎳ -> money
    ᎣᏍᏛ -> good
    ᎤᏲᎢ -> bad
    ᎡᎶᎯ -> Earth/world
    """
}

prompt_with_induced_vocab = """We are translating Cherokee to English. Here are a few vocabulary mappings from English to Cherokee to help you out:\n{vocab_mappings}\n\nPlease return the translation preceded by 'English: '\nInput: {input}"""

prompt_one_example_only = """We are translating Cherokee to English.\n\nPlease return the translation preceded by 'English: '\nInput: {input}"""

base_prompt_chr_en = {
    "system": PROBLEM_SOLVING_SYSPROMPT,
    "user": """We are translating Cherokee to English. Please return the translation preceded by 'Cherokee: '\nExamples:{few_shot_examples}\nInput: {input}""",
    "user_dict": """We are translating Cherokee to English. Please return the translation preceded by 'Cherokee: '\n{dict_entries}\n\nExamples:{few_shot_examples}\nInput: {input}"""
}

base_prompt_en_chr = {
    "system": PROBLEM_SOLVING_SYSPROMPT,
    "user": """We are translating English to Cherokee. Please return the translation preceded by 'English: '\nExamples:{few_shot_examples}\nInput: {input}""",
    "user_dict": """We are translating English to Cherokee. Please return the translation preceded by 'English: '\n{dict_entries}\n\nExamples:{few_shot_examples}\nInput: {input}"""
}

# TODO: note: the full grammar for real languages is too lengthy to fit into context.
# Not sure what to do about this at this point. Retrieving relevant rules depending on the input may be good

# TODO: I found a grammar reference at this website. Should get someone to check it's accurate
# https://www.cherokeelessons.com/content/Cherokee-Language-Grammar---Cherokee-Messenger-1844-1846/
prompt_with_true_grammar = {
    "system": GRAMMAR_USING_SYSPROMPT,
    "user": """Use this grammar to parse the input example to get the correct output.

    Cherokee alphabet:
    Ꭱ Ꭰ Ꮃ Ꮵ Ꮐ Ꮽ Ꮺ Ꮅ Ꮑ Ꮌ Ꭹ Ᏹ Ꮟ Ꮲ Ꭳ Ꮇ Ꮄ Ꭽ Ꮼ Ꮰ Ꮤ Ᏼ Ꮈ Ꭿ Ꮝ Ᏺ Ꮁ Ꭺ Ꮷ Ꮍ Ꮞ Ꮠ Ꮯ Ꮘ Ꮗ Ꮜ Ꮖ Ꮓ Ꭷ Ꮸ Ꮢ Ꮒ Ꭶ Ꮩ Ꭸ Ꮣ Ꭼ Ꮻ Ꭲ Ꭴ Ᏸ Ꮂ Ꮫ Ꭻ Ꮶ Ꮙ Ꮔ Ꮎ Ꮆ Ᏻ Ꮴ Ꮧ Ꮾ Ꮪ Ꮥ Ꮳ Ꭵ Ꮕ Ꮦ Ꮉ Ꮡ Ꮱ Ꭾ Ꮀ Ꮋ Ꮭ Ꮿ Ꮹ Ꮨ Ꮮ Ꮏ Ꮚ Ꮬ Ꮊ Ꮛ
    
    {grammar_fragment}

    {common_words}

    Examples:
    {few_shot_examples}

    Return the output preceded by 'Output:'
    Input: {input}
    """
}

prompt_with_true_grammar["user"] = prompt_with_true_grammar["user"].format(grammar_fragment="\n\n".join([cherokee_grammar_components["pronouns"], cherokee_grammar_components["verb_conjugation"]]), common_words=cherokee_grammar_components["common_words"], few_shot_examples="{few_shot_examples}", input="{input}")

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

few_shot_examples_chr_en_prompt = """Cherokee: {input}\nEnglish: {output}\n"""

few_shot_examples_en_chr_prompt = """English: {input}\nCherokee: {output}\n"""

prompt_for_vocab_induction = {
    "system": VOCAB_INDUCTION_SYSPROMPT,
    "user": """You are a linguist helping to document the Cherokee language. The following examples all contain the word {word}. Can you try to induce the likely mapping from the English word to the Cherokee word? Write it like this:\nEnglish word -> Cherokee word.

    {few_shot_examples}\nIf you don't think there's a clear mapping from this English word to a Cherokee word, just write "no mapping".
    
    Mapping:
    """,
}

prompt_for_dict_lookup = """Here are some vocabulary entries retrieved from a dictionary to help you translate. If a word has multiple possible translations, they will be separated with a comma.\n{matching_entries}"""
