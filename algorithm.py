from population import Population, Individual
import torch
import random
from torchvision.utils import save_image
from tqdm import tqdm
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
import numpy as np
from pymoo.util.randomized_argsort import randomized_argsort
from utils import set_seed_everything
import json
import os
from datetime import datetime
import pickle

set_seed_everything(222520691)

class GA:
    def __init__(self, sample_id,
                 n_iter, 
                 population, 
                 fitness,
                 tournament_size,
                 question,
                 answer,
                 fitness_statery,
                 pct_words_to_swap,
                 log_dir="ga_logs",
                 success_threshold=1.0):
        self.sample_id = sample_id
        self.n_iter = n_iter
        self.pop = population
        self.tournament_size = tournament_size
        self.fitness = fitness
        self.question = question
        self.answer = answer
        self.success_threshold = success_threshold
        self.fitness_statery = fitness_statery
        self.pct_words_to_swap = pct_words_to_swap
        
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.generation_logs = []
        self.best_individual = None
        self.best_fitness = None
        self.best_retri_score = None  
        self.best_reader_score = None  
        self.success_achieved = False
        self.success_generation = None
        self.adv_output = None

    def tournament_selection(self, fitness_array: np.ndarray, tournament_size: int = 4):
        selected_indices = []
        for j in range(0, len(fitness_array), tournament_size):
            group_fitness = fitness_array[j : j + tournament_size]
            best_in_group = j + np.argmin(group_fitness) 
            selected_indices.append(best_in_group)
        return selected_indices

    
    
    def solve_rule(self):
        P = self.pop.individuals
        P_fitness_weighted, P_retri_score, P_reader_score = self.fitness(
            question=self.question,
            contexts=[ind.get_perturbed_text() for ind in P],
            answer=self.answer
        )
        self.history = []

        for iter_idx in tqdm(range(self.n_iter), desc="GA Evolution"):
            O = []
            for _ in range(self.pop.pop_size // 2):
                parent_idx1, parent_idx2 = random.sample(range(self.pop.pop_size), 2)
                parent1, parent2 = P[parent_idx1], P[parent_idx2]
                offspring1, offspring2 = self.pop.crossover(parent1, parent2) # crossover
                offspring1 = self.pop.mutation(offspring1) # mutation
                offspring2 = self.pop.mutation(offspring2) # mutation
                O.extend([offspring1, offspring2])

            O_fitness_weighted, O_retri_score, O_reader_score = self.fitness(
                question=self.question,
                contexts=[ind.get_perturbed_text() for ind in O],
                answer=self.answer
            )
            # pool
            pool = P + O
            pool_fitness_weighted = np.concatenate([P_fitness_weighted, O_fitness_weighted], axis=0)
            pool_retri_score = np.concatenate([P_retri_score, O_retri_score], axis=0)
            pool_reader_score = np.concatenate([P_reader_score, O_reader_score], axis=0)

            # Find best individual in current generation (LOWEST weighted fitness = BEST)
            current_best_idx = np.argmin(pool_fitness_weighted)
            current_best_fitness = pool_fitness_weighted[current_best_idx]
            current_best_individual = pool[current_best_idx]
            current_best_retri_score = pool_retri_score[current_best_idx]  
            current_best_reader_score = pool_reader_score[current_best_idx] 
            
            self.best_individual = current_best_individual
            self.best_fitness = current_best_fitness
            self.best_retri_score = current_best_retri_score
            self.best_reader_score = current_best_reader_score
            
            self.history.append(np.stack([P_fitness_weighted, P_retri_score, P_reader_score], axis=0))
    
            # Selection for next generation
            pool_indices = np.arange(len(pool))
            selected_pool_index_parts = []
            for _ in range(self.tournament_size // 2):
                np.random.shuffle(pool_indices) # pool_indices
                selected = self.tournament_selection(pool_fitness_weighted[pool_indices]) # return seleted index from pool
                selected_pool_index_parts.append(pool_indices[selected]) # return 
            selected_pool_index = np.concatenate(selected_pool_index_parts)

            P = [pool[i] for i in selected_pool_index]
            P_fitness_weighted = pool_fitness_weighted[selected_pool_index]
            P_retri_score = pool_retri_score[selected_pool_index]
            P_reader_score = pool_reader_score[selected_pool_index]
        
        self.save_logs()
    
    def save_logs(self):
        score_log_file = os.path.join(self.log_dir, f"ga_{self.fitness_statery}_{self.pct_words_to_swap}_{self.sample_id}.pkl")
        text_log_file = os.path.join(self.log_dir, f"ga_{self.fitness_statery}_{self.pct_words_to_swap}_{self.sample_id}.txt")
        
        with open(score_log_file, 'wb') as f:
            pickle.dump(self.history, f)
        with open(text_log_file, "w") as f:
            f.write(self.best_individual.get_perturbed_text())

    
                        
        
        
class NSGAII:
    def __init__(self, sample_id, 
                 n_iter, 
                 population, 
                 fitness,
                 question,
                 answer,
                 fitness_statery,
                 pct_words_to_swap,
                 log_dir="nsgaii_logs",
                 success_threshold=1.0):
        self.sample_id = sample_id
        self.n_iter = n_iter
        self.pop = population
        self.fitness = fitness
        self.question = question
        self.answer = answer
        self.success_threshold = success_threshold
        self.fitness_statery = fitness_statery
        self.pct_words_to_swap = pct_words_to_swap
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.generation_logs = []
        self.best_individual = None
        self.best_fitness = None
        self.best_retri_score = None  
        self.best_reader_score = None  
        self.success_achieved = False
        self.success_generation = None
        self.adv_output = None
        self.log_file = os.path.join(log_dir, f"NSGAII_{fitness_statery}_{pct_words_to_swap}_{self.sample_id}.json")

        self.nds = NonDominatedSorting()
        
    def NSGA_selection(self, pool_fitness):
        nds = NonDominatedSorting()
        fronts = nds.do(pool_fitness, n_stop_if_ranked=self.pop.pop_size) # front ranked
        survivors = []
        for k, front in enumerate(fronts):
            crowding_of_front = self.calculating_crowding_distance(pool_fitness[front])
            sorted_indices = np.argsort(-crowding_of_front)
            front_sorted = [front[i] for i in sorted_indices]
            for idx in front_sorted:
                if len(survivors) < self.pop.pop_size:
                    survivors.append(idx)
                else:
                    break
            if len(survivors) >= self.pop.pop_size:
                break
        return survivors, fronts
    
    def calculating_crowding_distance(self, F):
        infinity = 1e+14

        n_points = F.shape[0]
        n_obj = F.shape[1]

        if n_points <= 2:
            return np.full(n_points, infinity)
        else:

            # sort each column and get index
            I = np.argsort(F, axis=0, kind='mergesort')

            # now really sort the whole array
            F = F[I, np.arange(n_obj)]

            # get the distance to the last element in sorted list and replace zeros with actual values
            dist = np.concatenate([F, np.full((1, n_obj), np.inf)]) - np.concatenate([np.full((1, n_obj), -np.inf), F])

            index_dist_is_zero = np.where(dist == 0)

            dist_to_last = np.copy(dist)
            for i, j in zip(*index_dist_is_zero):
                dist_to_last[i, j] = dist_to_last[i - 1, j]

            dist_to_next = np.copy(dist)
            for i, j in reversed(list(zip(*index_dist_is_zero))):
                dist_to_next[i, j] = dist_to_next[i + 1, j]

            # normalize all the distances
            norm = np.max(F, axis=0) - np.min(F, axis=0)
            norm[norm == 0] = np.nan
            dist_to_last, dist_to_next = dist_to_last[:-1] / norm, dist_to_next[1:] / norm

            # if we divided by zero because all values in one columns are equal replace by none
            dist_to_last[np.isnan(dist_to_last)] = 0.0
            dist_to_next[np.isnan(dist_to_next)] = 0.0

            # sum up the distance to next and last and norm by objectives - also reorder from sorted list
            J = np.argsort(I, axis=0)
            crowding = np.sum(dist_to_last[J, np.arange(n_obj)] + dist_to_next[J, np.arange(n_obj)], axis=1) / n_obj

        # replace infinity with a large number
        crowding[np.isinf(crowding)] = infinity
        return crowding

   
    def solve_rule(self):
        """
        Main NSGA-II evolution loop
        """
        P = self.pop.individuals
        P_retri_score, P_reader_score = self.fitness(
            question=self.question,
            contexts=[ind.get_perturbed_text() for ind in P],
            answer=self.answer
        )
        self.history = []

            
        for iter_idx in tqdm(range(self.n_iter), desc="NSGA-II Evolution"):
            # Generate offspring populatio
            O = []
            for _ in range(self.pop.pop_size // 2):
                parent_idx1, parent_idx2 = random.sample(range(self.pop.pop_size), 2)
                parent1, parent2 = P[parent_idx1], P[parent_idx2]
                offspring1, offspring2 = self.pop.crossover(parent1, parent2)  # crossover
                offspring1 = self.pop.mutation(offspring1)  # mutation
                offspring2 = self.pop.mutation(offspring2)  # mutation
                O.extend([offspring1, offspring2])

            # Evaluate offspring
            O_retri_score, O_reader_score = self.fitness(
                question=self.question,
                contexts=[ind.get_perturbed_text() for ind in O],
                answer=self.answer
            )

            # Create combined pool (P + O)
            pool = P + O
            pool_retri_score = np.concatenate([P_retri_score, O_retri_score], axis=0)
            pool_reader_score = np.concatenate([P_reader_score, O_reader_score], axis=0)
            pool_fitness = np.column_stack([pool_retri_score, pool_reader_score])
            
            
            # NSGA-II Selection for next generation
            selected_indices, fronts = self.NSGA_selection(pool_fitness)
            # Update population
            P = [pool[i] for i in selected_indices]
            P_retri_score = pool_retri_score[selected_indices]
            P_reader_score = pool_reader_score[selected_indices]
  
            
            rank_0_indices = fronts[0]  # Get indices of the first Pareto front
            rank_0_individuals = [pool[i] for i in rank_0_indices]
            rank_0_retri_scores = pool_retri_score[rank_0_indices]
            rank_0_reader_scores = pool_reader_score[rank_0_indices]    
            
            for i in range(rank_0_indices):
                print("\n" + "="*60)
                print("Individual: ", pool[i].get_perturbed_text())
                print("Retri Score: ", pool_retri_score[i])
                print("Reader Score: ", pool_reader_score[i])
            
            self.history.append(np.stack([rank_0_retri_scores, rank_0_reader_scores], axis=1))
            self.best_individual = rank_0_individuals
            self.best_retri_score = rank_0_retri_scores
            self.best_reader_score = rank_0_reader_scores
            

        
        self.save_logs()
                
                
    def save_logs(self):
        score_log_file = os.path.join(self.log_dir, f"ngsgaii_{self.fitness_statery}_{self.pct_words_to_swap}_{self.sample_id}.pkl")
        text_log_file = os.path.join(self.log_dir, f"nsgaii_{self.fitness_statery}_{self.pct_words_to_swap}_{self.sample_id}.txt")
        
        with open(score_log_file, 'wb') as f:
            pickle.dump(self.history, f)
            
        with open(text_log_file, "w", encoding="utf-8") as f:
            for ind in self.best_individual:
                f.write(ind.get_perturbed_text() + "\n")
