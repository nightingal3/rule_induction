from torch.utils.data import Sampler
import pickle
from typing import List, Optional, Callable, Tuple

class CurriculumSamplerSimple(Sampler):
    def __init__(self, difficulty_levels: List[float], threshold: Optional[int]):
        self.difficulty_levels = difficulty_levels
        self.threshold = min(difficulty_levels) if threshold is None else threshold
        self.indices = [i for i, d in enumerate(difficulty_levels) if d <= threshold]

    def step(self, step_size: int = 1):
        self.threshold += step_size
        self.indices = [i for i, d in enumerate(self.difficulty_levels) if d <= self.threshold]

    def __iter__(self):
        return iter(self.indices)
    
    def __len__(self):
        return len(self.indices)
    

## Utils for preproc
def compute_difficulty_word_len(sents: List[str]) -> int:
    return [len(sent.split(" ")) for sent in sents]

def make_compute_difficulty_freq(freq_dict_pkl: str = "") -> Callable:
    freq_dict = pickle.load(open(freq_dict_pkl, "rb"))
    
    def compute_difficulty_freq(sents: List[Tuple[str, str]]) -> int:
        freqs = []
        for sent in sents:
            freq = 0
            for word, tag in sent:
                if tag in ["NOUN", "VERB", "ADJ"]:
                    freq += freq_dict.get(f"{word}_{tag}", {"count": 0})["count"]
            
            freqs.append(-freq)
        
        return freqs
    
    return compute_difficulty_freq
