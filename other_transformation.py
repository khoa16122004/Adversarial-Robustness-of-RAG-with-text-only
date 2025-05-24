import random
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize, word_tokenize

class TextPerturber:
    def __init__(self, seed=42):
        random.seed(seed)

    def get_synonyms(self, word):
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                w = lemma.name().replace('_', ' ')
                if w.lower() != word.lower():
                    synonyms.add(w)
        return list(synonyms)

    def get_perturbed_sequences(self, original_text, indices_to_modify, num_words_to_swap, pop_size=5, seed=42):
        """
        Tạo các phiên bản của original_text bằng cách thay từ chỉ định bằng từ đồng nghĩa.
        """
        random.seed(seed)
        words = word_tokenize(original_text)
        candidates = []

        for _ in range(pop_size):
            new_words = words.copy()
            swapped = 0

            for idx in indices_to_modify:
                if idx >= len(new_words): continue
                synonyms = self.get_synonyms(new_words[idx])
                if synonyms:
                    new_words[idx] = random.choice(synonyms)
                    swapped += 1
                if swapped >= num_words_to_swap:
                    break

            candidates.append(' '.join(new_words))

        return candidates

    def get_shuffled_sentences(self, original_text, pop_size=5, seed=42):
        """
        Trả về các phiên bản của original_text với các câu bị đảo thứ tự.
        """
        random.seed(seed)
        sentences = sent_tokenize(original_text)
        candidates = []

        for _ in range(pop_size):
            shuffled = sentences.copy()
            random.shuffle(shuffled)
            candidates.append(' '.join(shuffled))

        return candidates
