from population import Population, Individual
import torch
import random
from torchvision.utils import save_image
from tqdm import tqdm
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
import numpy as np
from pymoo.util.randomized_argsort import randomized_argsort
from utils import set_seed_everything

set_seed_everything(222520691)

class GA:
    def __init__(self, n_iter, 
                 population, 
                 fitness,
                 tournament_size,
                 question,
                 answer):
        self.n_iter = n_iter
        self.pop = population
        self.tournament_size = tournament_size
        self.fitness = fitness
        self.question = question
        self.answer = answer

    def tournament_selection(self, fitness_array: np.ndarray, tournament_size: int = 4):
        selected_indices = []
        for j in range(0, len(fitness_array), tournament_size):
            group_fitness = fitness_array[j : j + tournament_size]
            best_in_group = j + np.argmax(group_fitness)  # hoặc np.argmin nếu cần minimize
            selected_indices.append(best_in_group)
        return selected_indices

    def solve_rule(self):
        P = self.pop.individuals
        P_fitness_weighted, P_score1, P_score2 = self.fitness(
            question=self.question,
            contexts=[ind.get_perturbed_text() for ind in P],
            answer=self.answer
        )

        best_individual = None
        best_fitness = None
        best_score1 = None
        best_score2 = None

        for iter_idx in tqdm(range(self.n_iter)):
            O = []
            for _ in range(self.pop.pop_size // 2):
                parent_idx1, parent_idx2 = random.sample(range(self.pop.pop_size), 2)
                parent1, parent2 = P[parent_idx1], P[parent_idx2]
                offspring1, offspring2 = self.pop.crossover(parent1, parent2)
                offspring1 = self.pop.mutation(offspring1)
                offspring2 = self.pop.mutation(offspring2)
                O.extend([offspring1, offspring2])

            O_fitness_weighted, O_score1, O_score2 = self.fitness(
                question=self.question,
                contexts=[ind.get_perturbed_text() for ind in O],
                answer=self.answer
            )

            pool = P + O
            pool_fitness_weighted = np.concatenate([P_fitness_weighted, O_fitness_weighted], axis=0)
            pool_score1 = np.concatenate([P_score1, O_score1], axis=0)
            pool_score2 = np.concatenate([P_score2, O_score2], axis=0)

            # Lấy cá thể tốt nhất dựa trên weighted sum fitness
            current_best_idx = np.argmax(pool_fitness_weighted)
            current_best_fitness = pool_fitness_weighted[current_best_idx]
            current_best_individual = pool[current_best_idx]
            current_best_score1 = pool_score1[current_best_idx]
            current_best_score2 = pool_score2[current_best_idx]

            if best_fitness is None or current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_individual = current_best_individual
                best_score1 = current_best_score1
                best_score2 = current_best_score2

            pool_indices = np.arange(len(pool))
            selected_pool_index_parts = []
            for _ in range(self.tournament_size // 2):
                np.random.shuffle(pool_indices)
                selected = self.tournament_selection(pool_fitness_weighted[pool_indices])
                selected_pool_index_parts.append(pool_indices[selected])
            selected_pool_index = np.concatenate(selected_pool_index_parts)

            P = [pool[i] for i in selected_pool_index]
            P_fitness_weighted = pool_fitness_weighted[selected_pool_index]
            P_score1 = pool_score1[selected_pool_index]
            P_score2 = pool_score2[selected_pool_index]
            print("Len Population: ", len(P))

        self.pop.individuals = P

        self.best_individual = best_individual
        self.best_fitness = best_fitness
        self.best_score1 = best_score1
        self.best_score2 = best_score2

