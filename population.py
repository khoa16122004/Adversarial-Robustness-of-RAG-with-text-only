import random
import copy
from textattack.shared import AttackedText
from typo_transformation import ComboTypoTransformation

class Individual:
    def __init__(self, original_text, 
                 replacement_words=None, 
                 modified_indices=None):
        self.attacked_text = AttackedText(original_text)
        self.replacement_words = replacement_words or []
        self.modified_indices = modified_indices or []

    def get_perturbed_text(self):
        if not self.modified_indices or not self.replacement_words:
            return self.attacked_text.text
        print(self.modified_indices)
        print(self.replacement_words)
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

    def _initialize_population(self):
        num_words_to_swap = max(int(self.pct_words_to_swap * len(self.indices_to_modify)), 1)
        per_words, per_words_indices = self.transformation.get_perturbed_sequences(
            self.original_text,
            self.indices_to_modify,
            num_words_to_swap,
            self.pop_size
        )
        print(per_words)
        print(per_words_indices)
        individuals = []
        for w, i in zip(per_words, per_words_indices):
            ind = Individual(self.original_text, w, i)
            individuals.append(ind)
        return individuals

    def crossover(self, ind1: Individual, ind2: Individual, crossover_prob=0.5):
        words1, indices1 = ind1.get_modified()
        words2, indices2 = ind2.get_modified()

        maintain_indices = list(set(indices1) & set(indices2)) # union
        ind1_only = list(set(indices1) - set(maintain_indices)) # 1\union
        ind2_only = list(set(indices2) - set(maintain_indices)) # 2\union

        cross_num = int(len(ind1_only)) * crossover_prob # chỉ riêng 1 * crossover_prob
        maintain_num = len(ind1_only) - cross_num

        if len(ind2_only) >= cross_num and len(ind1_only) >= maintain_num:
            maintain_indices += random.sample(ind1_only, k=maintain_num)
            maintain_words = [words1[indices1.index(idx)] for idx in maintain_indices]

            cross_indices = random.sample(ind2_only, k=cross_num)
            cross_words = [words2[indices2.index(idx)] for idx in cross_indices]

            child_indices = maintain_indices + cross_indices
            child_words = maintain_words + cross_words
            return Individual(self.original_text, child_words, child_indices)
        else:
            # Nếu không crossover được thì trả về bản sao ind1
            return copy.deepcopy(ind1)

    def mutation(self, ind: Individual, mutation_prob=0.4):
        words, indices = ind.get_modified()
        maintain_num = int(len(indices) * (1 - mutation_prob))
        if maintain_num > 0:
            maintain_indices = random.choices(indices, k=maintain_num)
            maintain_words = [words[indices.index(idx)] for idx in maintain_indices]
            modified_indices = list(set(self.indices_to_modify) - set(maintain_indices))
            num_words_to_swap = int(self.pct_words_to_swap * len(self.indices_to_modify) - len(maintain_words))
            per_words, per_words_indices = self.transformation.get_perturbed_sequences(
                AttackedText(self.original_text),
                modified_indices,
                num_words_to_swap,
                1
            )
            # Ghép phần giữ nguyên và phần mới mutate
            new_words = maintain_words + per_words[0]
            new_indices = maintain_indices + per_words_indices[0]
            return Individual(self.original_text, new_words, new_indices)
        else:
            return copy.deepcopy(ind)
        
def test_population():
    original_text = "The quick brown fox jumps over the lazy dog."
    indices_to_modify = list(range(len(original_text.split())))
    print(indices_to_modify)
    pop_size = 5

    # Dùng transformation thực tế
    transformation = ComboTypoTransformation()
    population = Population(original_text, pop_size, transformation, 
                            indices_to_modify, pct_words_to_swap=0.3)

    # print("Initial population:")
    # for ind in population.individuals:
    #     print(ind.get_perturbed_text())

    # Thử crossover
    print(population.individuals[0].get_perturbed_text())
    print(population.individuals[1].get_perturbed_text())
    child = population.crossover(population.individuals[0], population.individuals[1])
    print("\nCrossover result:")
    print(child.get_perturbed_text())

    # # Thử mutation
    # mutated = population.mutation(population.individuals[0])
    # print("\nMutation result:")
    # print(mutated.get_perturbed_text())

if __name__ == "__main__":
    test_population()