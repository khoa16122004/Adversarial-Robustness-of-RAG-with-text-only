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
        self.log_file = os.path.join(log_dir, f"ga_{fitness_statery}_{pct_words_to_swap}_{self.sample_id}.json")

    def tournament_selection(self, fitness_array: np.ndarray, tournament_size: int = 4):
        selected_indices = []
        for j in range(0, len(fitness_array), tournament_size):
            group_fitness = fitness_array[j : j + tournament_size]
            best_in_group = j + np.argmin(group_fitness) 
            selected_indices.append(best_in_group)
        return selected_indices

    def log_generation(self, generation, best_weighted_fitness, best_reader_score, best_retrieval_score, 
                      best_individual_text):

        generation_log = {
            "generation": generation,
            "best_weighted_fitness": float(best_weighted_fitness),
            "best_reader_score": float(best_reader_score),
            "best_retrieval_score": float(best_retrieval_score),
            "best_individual_text": best_individual_text,
            "success_achieved": bool(best_reader_score < self.success_threshold and 
                                   best_retrieval_score < self.success_threshold),
        }
        
        self.generation_logs.append(generation_log)
        
        if (best_reader_score < self.success_threshold and 
            best_retrieval_score < self.success_threshold and 
            not self.success_achieved):
            self.success_achieved = True
            self.success_generation = generation
            print(f"🎉 SUCCESS ACHIEVED at generation {generation}!")
            print(f"   Reader score: {best_reader_score:.6f}")
            print(f"   Retrieval score: {best_retrieval_score:.6f}")

    def save_logs(self):
        log_data = {
            "experiment_info": {
                "question": self.question,
                "answer": self.answer,
                "n_iterations": self.n_iter,
                "population_size": self.pop.pop_size,
                "tournament_size": self.tournament_size,
                "success_threshold": self.success_threshold,
            },
            "final_results": {
                "success_achieved": self.success_achieved,
                "success_generation": self.success_generation,
                "best_fitness": float(self.best_fitness) if self.best_fitness is not None else None,
                "best_reader_score": float(self.best_reader_score) if self.best_reader_score is not None else None,
                "best_retrieval_score": float(self.best_retri_score) if self.best_retri_score is not None else None,
                "best_individual_text": self.best_individual.get_perturbed_text() if self.best_individual else None,
                "adv_output": self.adv_output if self.adv_output is not None else None,
                "modified_info": self.best_individual.get_modified() if self.best_individual else None
            },
            "generation_logs": self.generation_logs
        }
        
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        
        print(f"📝 Logs saved to: {self.log_file}")

    
    
    def solve_rule(self):
        P = self.pop.individuals
        P_fitness_weighted, P_retri_score, P_reader_score = self.fitness(
            question=self.question,
            contexts=[ind.get_perturbed_text() for ind in P],
            answer=self.answer
        )

        print("🚀 Starting GA Evolution")
        print(f"   Population size: {self.pop.pop_size}")
        print(f"   Generations: {self.n_iter}")
        print(f"   Success threshold: {self.success_threshold}")
        print("="*60)

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
            
            # Update global best
            if self.best_fitness is None or current_best_fitness < self.best_fitness:
                self.best_fitness = current_best_fitness
                self.best_individual = current_best_individual
                self.best_retri_score = current_best_retri_score
                self.best_reader_score = current_best_reader_score
            

            # pop_stats = self.calculate_population_stats(pool_fitness_weighted, pool_score1, pool_score2)
            
            # Log generation data
            self.log_generation(
                generation=iter_idx,
                best_weighted_fitness=current_best_fitness,
                best_reader_score=current_best_reader_score,
                best_retrieval_score=current_best_retri_score,
                best_individual_text=current_best_individual.get_perturbed_text(),
            )

            # Print generation info
            print(f"\n📊 Generation {iter_idx}")
            print(f"   Best weighted fitness: {current_best_fitness:.6f}")
            print(f"   Best reader score: {current_best_reader_score:.6f}")
            print(f"   Best retrieval score: {current_best_retri_score:.6f}")
            print(f"   Success criteria met: {current_best_retri_score < self.success_threshold and current_best_reader_score < self.success_threshold}")
            
            # Optional: print generated answer for debugging
            try:
                generated_answer = self.fitness.reader.generate(self.question, [current_best_individual.get_perturbed_text()])
                if isinstance(generated_answer, list):
                    generated_answer = generated_answer[0] if generated_answer else "No answer"
                    self.adv_output = generated_answer
                print(f"   Generated answer: '{generated_answer.strip()}'")
            except Exception as e:
                print(f"   Generated answer: Error - {str(e)}")
            

            
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

        self.pop.individuals = P
        
        # Final results
        print("\n" + "="*60)
        print("🏁 GA Evolution Completed")
        print(f"   Final success status: {self.success_achieved}")
        if self.success_achieved:
            print(f"   Success achieved at generation: {self.success_generation}")
        print(f"   Best weighted fitness: {self.best_fitness:.6f}")
        print(f"   Best reader score: {self.best_reader_score:.6f}")
        print(f"   Best retrieval score: {self.best_retri_score:.6f}")
        print("="*60)
        
        # Save logs
        self.save_logs()
        
        return {
            "success": self.success_achieved,
            "success_generation": self.success_generation,
            "best_fitness": self.best_fitness,
            "best_reader_score": self.best_reader_score,
            "best_retrieval_score": self.best_retri_score,
            "best_individual": self.best_individual,
            "total_generations": iter_idx + 1,
            "log_file": self.log_file
        }

    def get_best_result(self):

        return {
            "individual": self.best_individual,
            "fitness": self.best_fitness,
            "reader_score": self.best_reader_score,
            "retrieval_score": self.best_retri_score,
            "text": self.best_individual.get_perturbed_text() if self.best_individual else None,
            "success": self.success_achieved
        }

    def print_summary(self):
        print("\n" + "="*60)
        print("📋 GA EVOLUTION SUMMARY")
        print("="*60)
        print(f"Question: {self.question}")
        print(f"Target Answer: {self.answer}")
        print(f"Success Threshold: {self.success_threshold}")
        print(f"Total Generations: {len(self.generation_logs)}")
        print()
        print("🎯 FINAL RESULTS:")
        print(f"   Success Achieved: {self.success_achieved}")
        if self.success_achieved:
            print(f"   Success Generation: {self.success_generation}")
        print(f"   Best Weighted Fitness: {self.best_fitness:.6f}")
        print(f"   Best Reader Score: {self.best_score1:.6f}")
        print(f"   Best Retrieval Score: {self.best_score2:.6f}")
        
        if self.best_individual:
            print(f"\n📝 BEST INDIVIDUAL:")
            print(f"   Text: {self.best_individual.get_perturbed_text()[:200]}...")
            
            # Generate answer with best individual
            try:
                generated = self.fitness.reader.generate(self.question, [self.best_individual.get_perturbed_text()])
                if isinstance(generated, list):
                    generated = generated[0] if generated else "No answer"
                print(f"   Generated Answer: '{generated.strip()}'")
            except Exception as e:
                print(f"   Generated Answer: Error - {str(e)}")
        
        print("="*60)
        
        
        
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
        

    def calculate_crowding_distance(self, objectives):
        """
        Calculate crowding distance for individuals based on their objectives
        """
        n_points, n_obj = objectives.shape
        
        if n_points <= 2:
            return np.full(n_points, np.inf)
        
        # Initialize crowding distance
        crowding_distance = np.zeros(n_points)
        
        # Calculate crowding distance for each objective
        for m in range(n_obj):
            # Sort points by objective m
            sorted_indices = np.argsort(objectives[:, m])
            
            # Set boundary points to infinity
            crowding_distance[sorted_indices[0]] = np.inf
            crowding_distance[sorted_indices[-1]] = np.inf
            
            # Calculate distance for intermediate points
            obj_range = objectives[sorted_indices[-1], m] - objectives[sorted_indices[0], m]
            
            if obj_range > 0:
                for i in range(1, n_points - 1):
                    crowding_distance[sorted_indices[i]] += (
                        objectives[sorted_indices[i + 1], m] - 
                        objectives[sorted_indices[i - 1], m]
                    ) / obj_range
        
        return crowding_distance

    def nsga_selection(self, population_indices, objectives, pop_size):
        """
        NSGA-II selection based on non-dominated sorting and crowding distance
        """
        # Perform non-dominated sorting
        fronts = self.nds.do(objectives, only_non_dominated_front=False)
        
        selected_indices = []
        
        # Select individuals from fronts
        for front in fronts:
            if len(selected_indices) + len(front) <= pop_size:
                # Add entire front
                selected_indices.extend(population_indices[front])
            else:
                # Calculate crowding distance for this front
                front_objectives = objectives[front]
                crowding_dist = self.calculate_crowding_distance(front_objectives)
                
                # Sort by crowding distance (descending)
                sorted_crowding_indices = np.argsort(-crowding_dist)
                
                # Select remaining individuals
                remaining = pop_size - len(selected_indices)
                selected_from_front = front[sorted_crowding_indices[:remaining]]
                selected_indices.extend(population_indices[selected_from_front])
                break
        
        return selected_indices

    def log_generation(self, generation, best_reader_score, best_retrieval_score, 
                      best_individual_text, all_retrieval_scores=None, all_reader_scores=None,
                      all_texts=None, population_type="current"):
        """
        Log generation data including all individual scores and Pareto front statistics
        """
        # Convert scores to lists for JSON serialization
        all_retrieval_list = all_retrieval_scores.tolist() if all_retrieval_scores is not None else []
        all_reader_list = all_reader_scores.tolist() if all_reader_scores is not None else []
        all_texts_list = all_texts if all_texts is not None else []
        
        # Calculate statistics for all scores
        score_stats = {}
        if all_retrieval_scores is not None and all_reader_scores is not None:
            score_stats = {
                "retrieval_scores": {
                    "mean": float(np.mean(all_retrieval_scores)),
                    "std": float(np.std(all_retrieval_scores)),
                    "min": float(np.min(all_retrieval_scores)),
                    "max": float(np.max(all_retrieval_scores)),
                    "median": float(np.median(all_retrieval_scores))
                },
                "reader_scores": {
                    "mean": float(np.mean(all_reader_scores)),
                    "std": float(np.std(all_reader_scores)),
                    "min": float(np.min(all_reader_scores)),
                    "max": float(np.max(all_reader_scores)),
                    "median": float(np.median(all_reader_scores))
                }
            }
        
        generation_log = {
            "generation": generation,
            "population_type": population_type,
            "best_reader_score": float(best_reader_score),
            "best_retrieval_score": float(best_retrieval_score),
            "best_individual_text": best_individual_text,
            "success_achieved": bool(best_reader_score < self.success_threshold and 
                                   best_retrieval_score < self.success_threshold),
            "score_statistics": score_stats,
            "all_scores": {
                "retrieval_scores": all_retrieval_list,
                "reader_scores": all_reader_list,
                "individual_texts": all_texts_list
            },
            "population_size": len(all_retrieval_list)
        }
        
        self.generation_logs.append(generation_log)
        
        if (best_reader_score < self.success_threshold and 
            best_retrieval_score < self.success_threshold and 
            not self.success_achieved):
            self.success_achieved = True
            self.success_generation = generation
            print(f"🎉 SUCCESS ACHIEVED at generation {generation}!")
            print(f"   Reader score: {best_reader_score:.6f}")
            print(f"   Retrieval score: {best_retrieval_score:.6f}")
    

    def save_logs(self):
        """
        Save experiment logs to JSON file
        """
        log_data = {
            "experiment_info": {
                "question": self.question,
                "answer": self.answer,
                "n_iterations": self.n_iter,
                "population_size": self.pop.pop_size,
                "success_threshold": self.success_threshold,
            },
            "final_results": {
                "success_achieved": self.success_achieved,
                "success_generation": self.success_generation,
                "best_fitness": float(self.best_fitness) if self.best_fitness is not None else None,
                "best_reader_score": float(self.best_reader_score) if self.best_reader_score is not None else None,
                "best_retrieval_score": float(self.best_retri_score) if self.best_retri_score is not None else None,
                "best_individual_text": self.best_individual.get_perturbed_text() if self.best_individual else None,
                "adv_output": self.adv_output if self.adv_output is not None else None,
                "modified_info": self.best_individual.get_modified() if self.best_individual else None
            },
            "generation_logs": self.generation_logs
        }
        
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        
        print(f"📝 Logs saved to: {self.log_file}")
        
        # Also save a summary CSV for easy analysis
        self.save_score_summary()

    def save_score_summary(self):
        """
        Save a CSV summary of all scores for easy analysis
        """
        summary_file = self.log_file.replace('.json', '_score_summary.csv')
        
        summary_data = []
        for log in self.generation_logs:
            gen = log['generation']
            pop_type = log.get('population_type', 'unknown')
            
            # Add each individual's scores as a row
            retrieval_scores = log['all_scores']['retrieval_scores']
            reader_scores = log['all_scores']['reader_scores']
            
            for i, (retri_score, reader_score) in enumerate(zip(retrieval_scores, reader_scores)):
                summary_data.append({
                    'generation': gen,
                    'population_type': pop_type,
                    'individual_id': i,
                    'retrieval_score': retri_score,
                    'reader_score': reader_score,
                    'is_best': (retri_score == log['best_retrieval_score'] and 
                               reader_score == log['best_reader_score'])
                })
        
        # Save to CSV
        import pandas as pd
        df = pd.DataFrame(summary_data)
        df.to_csv(summary_file, index=False)
        print(f"📊 Score summary saved to: {summary_file}")

    def calculate_pareto_stats(self, objectives):
        """
        Calculate Pareto front statistics
        """
        fronts = self.nds.do(objectives, only_non_dominated_front=False)
        
        pareto_front = fronts[0]
        pareto_objectives = objectives[pareto_front]
        
        return {
            "pareto_front_size": len(pareto_front),
            "total_fronts": len(fronts),
            "pareto_front_reader_scores": {
                "mean": float(np.mean(pareto_objectives[:, 1])),
                "std": float(np.std(pareto_objectives[:, 1])),
                "min": float(np.min(pareto_objectives[:, 1])),
                "max": float(np.max(pareto_objectives[:, 1]))
            },
            "pareto_front_retrieval_scores": {
                "mean": float(np.mean(pareto_objectives[:, 0])),
                "std": float(np.std(pareto_objectives[:, 0])),
                "min": float(np.min(pareto_objectives[:, 0])),
                "max": float(np.max(pareto_objectives[:, 0]))
            }
        }
    
    def greedy_selection(self, P_score1, P_score2):
        valid_indices = np.where(P_score1 < 1)[0]

        if len(valid_indices) > 0:
            min_index = valid_indices[np.argmin(P_score2[valid_indices])]
        else:
            min_index = np.argmin(P_score1)

        return min_index

    def update_global_best(self, current_best_retri_score, current_best_reader_score, current_best_individual):
        if self.best_retri_score is None:
            self.best_retri_score = current_best_retri_score
            self.best_reader_score = current_best_reader_score
            self.best_individual = current_best_individual
        
        else:
            pool_score1 = np.array([self.best_retri_score, current_best_retri_score])
            pool_score2 = np.array([self.best_reader_score, current_best_reader_score])
            
            if self.greedy_selection(pool_score1, pool_score2) == 1:
                self.best_retri_score = current_best_retri_score
                self.best_reader_score = current_best_reader_score
                self.best_individual = current_best_individual
   
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

        print("🚀 Starting NSGA-II Evolution")
        print(f"   Population size: {self.pop.pop_size}")
        print(f"   Generations: {self.n_iter}")
        print(f"   Success threshold: {self.success_threshold}")
        print("="*60)

        # Log initial population
        initial_best_idx = self.greedy_selection(P_retri_score, P_reader_score)
        self.log_generation(
            generation=-1,  # Use -1 to indicate initial population
            best_reader_score=P_reader_score[initial_best_idx],
            best_retrieval_score=P_retri_score[initial_best_idx],
            best_individual_text=P[initial_best_idx].get_perturbed_text(),
            all_retrieval_scores=P_retri_score,
            all_reader_scores=P_reader_score,
            all_texts=[ind.get_perturbed_text() for ind in P],
            population_type="initial"
        )

        for iter_idx in tqdm(range(self.n_iter), desc="NSGA-II Evolution"):
            # Generate offspring population
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

            # Log offspring population
            offspring_best_idx = self.greedy_selection(O_retri_score, O_reader_score)
            self.log_generation(
                generation=iter_idx,
                best_reader_score=O_reader_score[offspring_best_idx],
                best_retrieval_score=O_retri_score[offspring_best_idx],
                best_individual_text=O[offspring_best_idx].get_perturbed_text(),
                all_retrieval_scores=O_retri_score,
                all_reader_scores=O_reader_score,
                all_texts=[ind.get_perturbed_text() for ind in O],
                population_type="offspring"
            )

            # Create combined pool (P + O)
            pool = P + O
            pool_retri_score = np.concatenate([P_retri_score, O_retri_score], axis=0)
            pool_reader_score = np.concatenate([P_reader_score, O_reader_score], axis=0)

            # Log combined pool
            pool_best_idx = self.greedy_selection(pool_retri_score, pool_reader_score)
            self.log_generation(
                generation=iter_idx,
                best_reader_score=pool_reader_score[pool_best_idx],
                best_retrieval_score=pool_retri_score[pool_best_idx],
                best_individual_text=pool[pool_best_idx].get_perturbed_text(),
                all_retrieval_scores=pool_retri_score,
                all_reader_scores=pool_reader_score,
                all_texts=[ind.get_perturbed_text() for ind in pool],
                population_type="combined_pool"
            )
            
            # Update global best
            current_best_idx = self.greedy_selection(pool_retri_score, pool_reader_score)
            current_best_retri_score = pool_retri_score[current_best_idx]
            current_best_reader_score = pool_reader_score[current_best_idx]
            current_best_individual = pool[current_best_idx]
            
            # update global test
            self.update_global_best(current_best_retri_score, current_best_reader_score, current_best_individual)          
            # score1 = retrieval_score, score2 = reader_score
            objectives = np.column_stack([pool_retri_score, pool_reader_score])
            pareto_stats = self.calculate_pareto_stats(objectives)

            # Print generation info
            print(f"\n📊 Generation {iter_idx}")
            print(f"   Pareto front size: {pareto_stats['pareto_front_size']}")
            print(f"   Total fronts: {pareto_stats['total_fronts']}")
            print(f"   Best combined pool scores: R={current_best_retri_score:.6f}, G={current_best_reader_score:.6f}")
            
            # Optional: print generated answer for debugging
            try:
                generated_answer = self.fitness.reader.generate(self.question, [current_best_individual.get_perturbed_text()])
                if isinstance(generated_answer, list):
                    generated_answer = generated_answer[0] if generated_answer else "No answer"
                    self.adv_output = generated_answer
                print(f"   Generated answer: '{generated_answer.strip()}'")
            except Exception as e:
                print(f"   Generated answer: Error - {str(e)}")
            
            # NSGA-II Selection for next generation
            pool_indices = np.arange(len(pool))
            selected_indices = self.nsga_selection(pool_indices, objectives, self.pop.pop_size)
            
            # Update population
            P = [pool[i] for i in selected_indices]
            P_retri_score = pool_retri_score[selected_indices]
            P_reader_score = pool_reader_score[selected_indices]

            # Log selected population for next generation
            selected_best_idx = self.greedy_selection(P_retri_score, P_reader_score)
            self.log_generation(
                generation=iter_idx,
                best_reader_score=P_reader_score[selected_best_idx],
                best_retrieval_score=P_retri_score[selected_best_idx],
                best_individual_text=P[selected_best_idx].get_perturbed_text(),
                all_retrieval_scores=P_retri_score,
                all_reader_scores=P_reader_score,
                all_texts=[ind.get_perturbed_text() for ind in P],
                population_type="selected_next_gen"
            )

        # Update population
        self.pop.individuals = P
        
        # Final results
        print("\n" + "="*60)
        print("🏁 NSGA-II Evolution Completed")
        print(f"   Final success status: {self.success_achieved}")
        if self.success_achieved:
            print(f"   Success achieved at generation: {self.success_generation}")
        print(f"   Best reader score: {self.best_reader_score:.6f}")
        print(f"   Best retrieval score: {self.best_retri_score:.6f}")
        print("="*60)
        
        # Save logs
        self.save_logs()
        
        return {
            "success": self.success_achieved,
            "success_generation": self.success_generation,
            "best_reader_score": self.best_reader_score,
            "best_retrieval_score": self.best_retri_score,
            "best_individual": self.best_individual,
            "total_generations": iter_idx + 1,
            "log_file": self.log_file
        }

    def get_best_result(self):
        """
        Return best result
        """
        return {
            "individual": self.best_individual,
            "fitness": self.best_fitness,
            "reader_score": self.best_reader_score,
            "retrieval_score": self.best_retri_score,
            "text": self.best_individual.get_perturbed_text() if self.best_individual else None,
            "success": self.success_achieved
        }

    def get_pareto_front(self):
        """
        Get current Pareto front individuals
        """
        if not self.pop.individuals:
            return []
        
        # Evaluate current population
        score1, score2 = self.fitness(
            question=self.question,
            contexts=[ind.get_perturbed_text() for ind in self.pop.individuals],
            answer=self.answer
        )
        
        # Prepare objectives
        objectives = np.column_stack([score1, score2])
        
        # Get Pareto front
        fronts = self.nds.do(objectives, only_non_dominated_front=True)
        pareto_indices = fronts[0]
        
        pareto_individuals = []
        for idx in pareto_indices:
            pareto_individuals.append({
                "individual": self.pop.individuals[idx],
                "reader_score": score2[idx],
                "retrieval_score": score1[idx],
                "text": self.pop.individuals[idx].get_perturbed_text()
            })
        
        return pareto_individuals

    def print_summary(self):
        """
        Print evolution summary
        """
        print("\n" + "="*60)
        print("📋 NSGA-II EVOLUTION SUMMARY")
        print("="*60)
        print(f"Question: {self.question}")
        print(f"Target Answer: {self.answer}")
        print(f"Success Threshold: {self.success_threshold}")
        print(f"Total Generations: {len([log for log in self.generation_logs if log.get('population_type') == 'selected_next_gen'])}")
        print()
        print("🎯 FINAL RESULTS:")
        print(f"   Success Achieved: {self.success_achieved}")
        if self.success_achieved:
            print(f"   Success Generation: {self.success_generation}")
        print(f"   Best Reader Score: {self.best_reader_score:.6f}")
        print(f"   Best Retrieval Score: {self.best_retri_score:.6f}")
        
        if self.best_individual:
            print(f"\n📝 BEST INDIVIDUAL:")
            best_text = self.best_individual.get_perturbed_text()
            print(f"   Text: {best_text[:200]}{'...' if len(best_text) > 200 else ''}")
            
            # Generate answer with best individual
            try:
                generated = self.fitness.reader.generate(self.question, [self.best_individual.get_perturbed_text()])
                if isinstance(generated, list):
                    generated = generated[0] if generated else "No answer"
                print(f"   Generated Answer: '{generated.strip()}'")
            except Exception as e:
                print(f"   Generated Answer: Error - {str(e)}")
        
        # Print Pareto front info
        try:
            pareto_front = self.get_pareto_front()
            print(f"\n🔄 PARETO FRONT:")
            print(f"   Size: {len(pareto_front)}")
            if pareto_front:
                print(f"   Reader Score Range: [{min(p['reader_score'] for p in pareto_front):.6f}, {max(p['reader_score'] for p in pareto_front):.6f}]")
                print(f"   Retrieval Score Range: [{min(p['retrieval_score'] for p in pareto_front):.6f}, {max(p['retrieval_score'] for p in pareto_front):.6f}]")
        except Exception as e:
            print(f"   Pareto Front: Error calculating - {str(e)}")
        
        print("="*60)