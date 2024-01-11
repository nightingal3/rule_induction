GRAMMAR_INDUCTION_SYSPROMPT = "You are a linguist. Your task is to deduce a grammar for the target language based on the examples given. The grammar will be used to translate between the target language and English."
PROBLEM_SOLVING_SYSPROMPT = "You are an expert translator. You are translating between English and another language. Use the examples to guide your thinking."
GRAMMAR_USING_SYSPROMPT = "You are an expert translator. You are translating between English and another language. You are provided with a grammar, which you should use to parse the examples and translate them."
VOCAB_INDUCTION_SYSPROMPT = "You are a linguist. Your task is to deduce a vocabulary for the target language based on the examples given. The vocabulary will be used to translate between the given language and English."
# grammar comes from this page: https://www.cherokeelessons.com/content/Cherokee-Language-Grammar---Cherokee-Messenger-1844-1846/
# nouns come from this word list: https://language.cherokee.org/word-list/

base_prompt = {
    "system": PROBLEM_SOLVING_SYSPROMPT,
    "user": """We are translating {src_lang} to {tgt_lang}. This is a linguistic puzzle. Additionally, here's some extra information about the language: {meta}\nPlease return the translation preceded by '{tgt_lang}: '\nExamples:{few_shot_examples}\nInput: {input}"""
}

few_shot_examples_src_tgt_prompt = """{src_lang}: {input}\n{tgt_lang}: {output}\n"""

vocab_induction_selected_examples_prompt = """We're learning to translate {src_lang} to {tgt_lang}. Let's try to find out what this means in {tgt_lang}:\nInput: {input}\n\nRelated sentences:\n{demonstrations}\nPlease return the translation preceded by '{tgt_lang}: '\nGiven this sentence, and the fact that {src_lang} has a {word_order} word order, what does '{input}' likely mean? Write your answer like this: {src_lang} word -> {tgt_lang} word."""

vocab_induction_final_prompt = """We are translating {src_lang} to {tgt_lang}. This is a linguistic puzzle. Additionally, here's some extra information about the language: {meta}\n\nExamples:{demonstrations}\nHere is some induced vocabulary to help you:\n{rules}\n{src_lang}: {input}""" 

word_order_guess_prompt = """We are documenting {other_lang}. This is a linguistic puzzle. Please output the word order you think that this language follows, based on these few shot examples: {few_shot_examples}. Please write your answer like this: 'word order: SVO'. If you think that the word order is flexible, please write 'word order: flexible'"""
ground_truth_lang_features = {
    "basque": "SOV",
    "english": "SVO",
    "norwegian": "SVO",
    "blackfoot": "flexible",
    "wambaya": "flexible, with a preference for verb-object ordering",
    "yonggom": "SOV",
    "chickasaw": "SOV",
    "euskara": "SOV",
    "luiseño": "flexible, with a preference for subject-verb ordering",
    "dyirbal": "flexible",
    "madak": "flexible, as it is an agglutinative language."
}