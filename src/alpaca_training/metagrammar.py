from nltk import PCFG, Nonterminal
from nltk.parse import pchart
import random

# TODO: finish this class later
class SynchronousGrammar:
    def __init__(self, word_mappings, start_symbol="S"):
        self.start_symbol = start_symbol
        self.word_mappings = word_mappings
        self.grammar = self._build_grammar()

    def _build_grammar(self):
        productions = []
        for lhs, rhs in self.word_mappings.items():
            productions.append((Nonterminal(lhs), tuple(rhs)))
        return PCFG(Nonterminal(self.start_symbol), productions)
