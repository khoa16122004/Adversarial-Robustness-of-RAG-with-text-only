import random
import copy
from textattack.shared import AttackedText
from typo_transformation import ComboTypoTransformation
from utils import set_seed_everything

set_seed_everything(222520691)

class Individual:
    def __init__(self, original_text, 
                 replacement_words=None, 
                 modified_indices=None):
        self.attacked_text = AttackedText(original_text)
        self.replacement_words = replacement_words or []
        self.modified_indices = modified_indices or []
        # self.original_splits = original_text.split()

    def get_perturbed_text(self):
        if not self.modified_indices or not self.replacement_words:
            return self.attacked_text.text
        return self.attacked_text.replace_words_at_indices(self.modified_indices, self.replacement_words).text

    def set_modified(self, words, indices):
        self.replacement_words = words
        self.modified_indices = indices

    def get_modified(self):
        return self.replacement_words, self.modified_indices

class Population:
    def __init__(self, original_text, pop_size, transformation, indices_to_modify, pct_words_to_swap=0.1):
        self.original_text = original_text
        self.pop_size = pop_size
        self.transformation = transformation
        self.indices_to_modify = indices_to_modify
        self.pct_words_to_swap = pct_words_to_swap # percentage words / sentence to swap
        self.individuals = self._initialize_population()
        self.original_splits = self.original_text.split()

    def _initialize_population(self):
        num_words_to_swap = max(int(self.pct_words_to_swap * len(self.indices_to_modify)), 1)
        per_words, per_words_indices = self.transformation.get_perturbed_sequences(
            self.original_text,
            self.indices_to_modify,
            num_words_to_swap,
            self.pop_size
        )
        individuals = []
        for w, i in zip(per_words, per_words_indices):
            ind = Individual(self.original_text, w, i)
            individuals.append(ind)
        return individuals

    def crossover(self, ind1: Individual, ind2: Individual, crossover_prob=0.5):
        words1, indices1 = ind1.get_modified()
        words2, indices2 = ind2.get_modified()

        set1, set2 = set(indices1), set(indices2)
        giao_set = set1 & set2
        # print("Giao set: ", giao_set)
        ind1_only = list(set1 - giao_set)
        ind2_only = list(set2 - giao_set)
        num_change = max(len(indices1), len(indices2))

        num_cross = int(crossover_prob * (num_change - len(giao_set)))
       

        if num_cross == 0:
            return copy.deepcopy(ind1), copy.deepcopy(ind2)

        cross1 = random.sample(ind1_only, min(num_cross, len(ind1_only)))
        cross2 = random.sample(ind2_only, min(num_cross, len(ind2_only)))

        child1_indices = list(giao_set) + [i for i in ind1_only if i not in cross1] + cross2
        # print(child1_indices)
        child1_words = [words1[indices1.index(i)] for i in giao_set] + \
                    [words1[indices1.index(i)] for i in ind1_only if i not in cross1] + \
                    [words2[indices2.index(i)] for i in cross2]

        child2_indices = list(giao_set) + [i for i in ind2_only if i not in cross2] + cross1
        # print(child2_indices)
        child2_words = [words2[indices2.index(i)] for i in giao_set] + \
                    [words2[indices2.index(i)] for i in ind2_only if i not in cross2] + \
                    [words1[indices1.index(i)] for i in cross1]

        return Individual(self.original_text, child1_words, child1_indices), \
            Individual(self.original_text, child2_words, child2_indices)

    def mutation(self, ind: Individual, mutation_prob=0.3):
            words, indices = ind.get_modified()
            if random.random() < mutation_prob:
                # Chọn ngẫu nhiên một index từ toàn bộ indices_to_modify
                mutate_idx = random.choice(indices)
                word_pos = indices.index(mutate_idx)

                typo_candidates = self.transformation.get_replacement_words(self.original_splits[mutate_idx])
                if not typo_candidates:
                    return copy.deepcopy(ind)
                new_word = random.choice(typo_candidates)
                new_words = words.copy()
                if word_pos < len(new_words):
                    new_words[word_pos] = new_word
                else:
                    new_words.append(new_word)
                return Individual(self.original_text, new_words, indices)
            else:
                return copy.deepcopy(ind)

def create_population(original_text, args):
    transformation = ComboTypoTransformation()
    indices_to_modify = list(range(len(original_text.split())))
    return Population(original_text, 
                      args.pop_size, 
                      transformation,
                      indices_to_modify, 
                      args.pct_words_to_swap)
    
     
# def test_population():
#     original_text = "The quick brown fox jumps over the lazy dog."
#     indices_to_modify = list(range(len(original_text.split())))
#     pop_size = 5

#     # Dùng transformation thực tế
#     transformation = ComboTypoTransformation()
#     population = Population(original_text, pop_size, transformation, 
#                             indices_to_modify, pct_words_to_swap=0.5)

#     # print("Initial population:")
#     # for ind in population.individuals:
#     #     print(ind.get_perturbed_text())

#     # Thử crossover
#     # print("First: ", population.individuals[0].get_perturbed_text())
#     # print("Second: ", population.individuals[1].get_perturbed_text())
#     # child1, child2 = population.crossover(population.individuals[0], population.individuals[1])
#     # print("\nCrossover result:")
#     # print(child1.get_perturbed_text())
#     # print(child2.get_perturbed_text())

#     # Thử mutation
#     print("First: ", population.individuals[0].get_perturbed_text())

#     mutated = population.mutation(population.individuals[0])
#     print("\nMutation result:")
#     print(mutated.get_perturbed_text())

# if __name__ == "__main__":
#     test_population()