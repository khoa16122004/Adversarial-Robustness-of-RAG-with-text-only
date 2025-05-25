import os
import copy
import random
import numpy as np
from utils import set_seed_everything
from nltk.corpus import wordnet

set_seed_everything(222520691)

class BaseTypoTransformation:
    def __init__(self):
        self.typos = {}
        self.NN = {}
        natural_path = os.path.join("noise", "en.natural")
        key_path = os.path.join("noise", "en.key")
        if os.path.exists(natural_path):
            for line in open(natural_path):
                line = line.strip().split()
                self.typos[line[0]] = line[1:]
        if os.path.exists(key_path):
            for line in open(key_path):
                line = line.split()
                self.NN[line[0]] = line[1:]

    def get_replacement_words(self, word):
        raise NotImplementedError("Implement in subclass")

class KeyboardTypoTransformation(BaseTypoTransformation):
    def get_replacement_words(self, word):
        words = []
        chars = list(word)
        for i, char in enumerate(chars):
            if char in self.NN:
                for w in self.NN[char.lower()]:
                    replace_word = copy.deepcopy(chars)
                    replace_word[i] = w
                    words.append(''.join(replace_word))
            elif char.lower() in self.NN:
                for w in self.NN[char.lower()]:
                    replace_word = copy.deepcopy(chars)
                    replace_word[i] = w.upper()
                    words.append(''.join(replace_word))
        return list(set(words))

class NaturalTypoTransformation(BaseTypoTransformation):
    def get_replacement_words(self, word):
        return self.typos.get(word, [])

class TruncateTypoTransformation(BaseTypoTransformation):
    def get_replacement_words(self, word, minlen=3, cutoff=3):
        words = []
        chars = list(word)
        tmp_cutoff = cutoff
        while len(chars) > minlen and tmp_cutoff > 0:
            chars = chars[:-1]
            tmp_cutoff -= 1
            words.append(''.join(chars))
        chars = list(word)
        tmp_cutoff = cutoff
        while len(chars) > minlen and tmp_cutoff > 0:
            chars = chars[1:]
            tmp_cutoff -= 1
            words.append(''.join(chars))
        return words

class InnerSwapTypoTransformation(BaseTypoTransformation):
    def get_replacement_words(self, word):
        def __shuffle_string__(_word, _seed=42):
            chars = list(_word)
            if _seed is not None:
                np.random.seed(_seed)
            np.random.shuffle(chars)
            return ''.join(chars)
        words = []
        if len(word) <= 3:
            return words
        tries = 0
        min_perturb = min(int(len(word)*0.4),2)
        while tries < 5:
            tries += 1
            start = random.randrange(1, len(word)-min_perturb+1)
            first, mid, last = word[0:start], word[start:start+min_perturb], word[start+min_perturb:]
            words.append(first + __shuffle_string__(mid) + ''.join(last))
        return list(set(words))


class SynonymsTransformation:
    def get_perturbed_sequences(self, original_text, indices_to_modify, num_words_to_swap, pop_size=5, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        words = original_text.split()
        per_words = []
        per_words_indices = []
        for _ in range(pop_size):
            chosen_indices = random.sample(indices_to_modify, num_words_to_swap) # nên là ko chọn lại
            new_words = []            
            for idx in chosen_indices:
                typo_candidates = self.get_replacement_words(words[idx])
                replace_word = random.choice(typo_candidates)
                if not replace_word:
                    print("Typo candidate is empty: ", typo_candidates)
                    raise
                new_words.append(random.choice(typo_candidates))
            per_words.append(new_words)
            per_words_indices.append(chosen_indices)
        return per_words, per_words_indices
    
    
    def get_replacement_words(self, word):
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                w = lemma.name().replace('_', ' ')
                if w.lower() != word.lower():
                    synonyms.add(w)
        return list(synonyms)

class ComboTypoTransformation(BaseTypoTransformation):
    def get_perturbed_sequences(self, original_text, 
                                indices_to_modify, 
                                num_words_to_swap,
                                pop_size=5, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        words = original_text.split()
        per_words = []
        per_words_indices = []
        for _ in range(pop_size):
            chosen_indices = random.sample(indices_to_modify, num_words_to_swap) # nên là ko chọn lại
            new_words = []            
            for idx in chosen_indices:
                typo_candidates = self.get_replacement_words(words[idx])
                if not typo_candidates:
                    typo_candidates = [words[idx]]
                replace_word = random.choice(typo_candidates)
                if not replace_word:
                    print("Typo candidate is empty: ", typo_candidates)
                    raise
                new_words.append(random.choice(replace_word))
            per_words.append(new_words)
            per_words_indices.append(chosen_indices)
        return per_words, per_words_indices

    def get_replacement_words(self, word):
        words = []
        words += KeyboardTypoTransformation().get_replacement_words(word)
        words += NaturalTypoTransformation().get_replacement_words(word)
        words += TruncateTypoTransformation().get_replacement_words(word)
        words += InnerSwapTypoTransformation().get_replacement_words(word)
        return list(set(words))


if __name__ == "__main__":
    transformation = SynonymsTransformation()
    
    print(transformation.get_replacement_words("quick"))