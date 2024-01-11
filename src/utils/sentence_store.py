from typing import Any, Optional, Tuple, List
from collections import defaultdict

class SentenceStore:
    def __init__(self):
        self.categories = defaultdict(list)
        self.word_list = defaultdict(list)
        self.unique_nouns = set()
        self.unique_verbs = set()
        self.unique_adjectives = set()
        self.contrast_categories = [("singular", "plural"), ("present", "past")]

    def __str__(self):
        return f'SentenceStore({self.categories.keys()})'

    def __len__(self):
        return sum([len(self.categories[x]) for x in self.categories])
    
    def add_sentence(self, sentence_pair: Tuple[str, str], categories: set, unique_nouns: set = None, unique_verbs: set = None, unique_adjectives: set = None):
        sentence_en, sentence_en_lemma, sentence_other = sentence_pair
        for cat in categories:
            self.categories[cat].append(sentence_pair)
        for word in sentence_en_lemma.split():
            clean_word = word.strip('.,?!').lower()
            self.word_list[clean_word].append(sentence_pair)

    def get_sentences(self, lemma_word: Optional[str] = None, category: Optional[str] = None):
        if lemma_word is None and category is None:
            raise ValueError('Must provide either word or category')
        if lemma_word and not category:
            return self.word_list[lemma_word]
        if category and not lemma_word:
            return self.categories[category]
        else:
            return [x for x in self.categories[category] if lemma_word in x.split()]
        
    def get_contrast_sets(self) -> List[List]:
        # Get all sentences with contrasting features (e.g. singular vs plural)
        contrastive_sentences = {}
        for cat1, cat2 in self.contrast_categories:
            cat1_examples = self.get_sentences(category=cat1)
            cat2_examples = self.get_sentences(category=cat2)
            if len(cat1_examples) > 0 and len(cat2_examples) > 0:
                contrastive_sentences[(cat1, cat2)] = [cat1_examples, cat2_examples]
        
        # TODO: maybe add examples of the same word with diverse contexts
        return contrastive_sentences
