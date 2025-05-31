import random
import copy
from textattack.shared import AttackedText
from typo_transformation import ComboTypoTransformation
from utils import set_seed_everything, split, find_answer
import re
set_seed_everything(222520691)




class Individual:
    def __init__(self, 
                 original_text: str,
                 answer: str,
                 original_text_split: list,
                 answer_position: list,
                 replacement_words: list,
                 modified_indices: list):
        self.original_text = original_text
        self.answer = answer
        self.original_text_split = original_text_split # text split
        self.answer_position = answer_position  
        # adv things
        self.replacement_words = replacement_words
        self.modified_indices = modified_indices # 
        self.attacked_text = AttackedText(original_text)
        

    def get_perturbed_text(self):
        if not self.modified_indices or not self.replacement_words:
            return self.attacked_text.text
        return self.attacked_text.replace_words_at_indices(self.modified_indices, self.replacement_words).text

    def set_modified(self, words, indices):
        self.replacement_words = words
        self.modified_indices = indices

    def get_modified(self):
        return copy.deepcopy(self.replacement_words), copy.deepcopy(self.modified_indices)
    
        
    

class Population:
    def __init__(self, 
                 original_text, 
                 answer,
                 pop_size, 
                 pct_words_to_swap=0.1):
        
        self.original_text = original_text
        self.answer = answer
        self.pop_size = pop_size
        self.pct_words_to_swap = pct_words_to_swap 
        
        self.original_text_split = [s.lower() for s in split(original_text)]
        self.answer_split = [s.lower() for s in split(answer)]
        self.answer_position_indices = find_answer(self.original_text_split, self.answer_split)
        self.indices_to_modify = [
            i for i in range(len(self.original_text_split))
            if i not in self.answer_position_indices
        ]
        
        # transformation
        self.transformation = ComboTypoTransformation()

        self.num_words_to_swap = max(int(self.pct_words_to_swap * len(self.indices_to_modify)), 1)
        self._initialize_population()
        
    def _initialize_population(self):
        per_words, per_words_indices = self.transformation.get_perturbed_sequences(
            original_text_split=self.original_text_split,
            indices_to_modify=self.indices_to_modify,
            num_words_to_swap=self.num_words_to_swap,
            pop_size=self.pop_size,
        )
        
        individuals = []
        for w, i in zip(per_words, per_words_indices):
            ind = Individual(original_text=self.original_text, 
                            answer=self.answer,
                            original_text_split=self.original_text_split,
                            answer_position=self.answer_position_indices,
                            replacement_words=w,
                            modified_indices=i)
            
            individuals.append(ind)
        
        self.individuals = individuals            

    def crossover(self, ind1: Individual, ind2: Individual):
        words1, indices1 = ind1.get_modified()
        words2, indices2 = ind2.get_modified()

        off1 = copy.deepcopy(ind1)
        off2 = copy.deepcopy(ind2)
        
        set1, set2 = set(indices1), set(indices2)
        giao_set = set1 & set2
        ind1_only = list(set1 - giao_set)
        ind2_only = list(set2 - giao_set)
        num_change = max(len(indices1), len(indices2))

        num_cross = (num_change - len(giao_set))
        # print("Num cross: ", num_cross)

        if num_cross == 0:
            return copy.deepcopy(ind1), copy.deepcopy(ind2)

        cross1 = random.sample(ind1_only, min(num_cross, len(ind1_only)))
        cross2 = random.sample(ind2_only, min(num_cross, len(ind2_only)))

        child1_indices = list(giao_set) + [i for i in ind1_only if i not in cross1] + cross2
        child1_words = [words1[indices1.index(i)] for i in giao_set] + \
                    [words1[indices1.index(i)] for i in ind1_only if i not in cross1] + \
                    [words2[indices2.index(i)] for i in cross2]

        child2_indices = list(giao_set) + [i for i in ind2_only if i not in cross2] + cross1
        child2_words = [words2[indices2.index(i)] for i in giao_set] + \
                    [words2[indices2.index(i)] for i in ind2_only if i not in cross2] + \
                    [words1[indices1.index(i)] for i in cross1]

        off1.set_modified(child1_words, child1_indices)
        off2.set_modified(child2_words, child2_indices)
        
        return off1, off2

    def mutation(self, ind: Individual, mutation_prob=0.3):
            words, indices = ind.get_modified( )
            if random.random() < mutation_prob:
                new_words = None
                while not new_words:
                    new_idx = random.choice([i for i in self.indices_to_modify if i not in indices])
                # Ensure the new index is not in the current indices
                    new_words = self.transformation.get_replacement_words(self.original_text_split[new_idx])
                mutate_idx = random.choice(indices)
                pos_idx = indices.index(mutate_idx)
                indices.pop(pos_idx)
                words.pop(pos_idx)
                
                indices.append(new_idx)
                words.append(random.choice(new_words))
                
                ind.set_modified(
                    words,
                    indices
                )
            return copy.deepcopy(ind)
        

