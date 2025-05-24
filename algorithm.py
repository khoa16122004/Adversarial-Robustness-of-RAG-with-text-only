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
    def __init__(self, n_iter, 
                 population, 
                 fitness,
                 tournament_size,
                 question,
                 answer,
                 log_dir="ga_logs",
                 success_threshold=1.0):
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
        self.best_score1 = None  
        self.best_score2 = None  
        self.success_achieved = False
        self.success_generation = None
        self.adv_output = None
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"ga_log_{timestamp}.json")

    def tournament_selection(self, fitness_array: np.ndarray, tournament_size: int = 4):
        selected_indices = []
        for j in range(0, len(fitness_array), tournament_size):
            group_fitness = fitness_array[j : j + tournament_size]
            best_in_group = j + np.argmin(group_fitness)  # hoặc np.argmin nếu cần minimize
            selected_indices.append(best_in_group)
        return selected_indices

    def log_generation(self, generation, best_weighted_fitness, best_reader_score, best_retrieval_score, 
                      best_individual_text, population_stats):

        generation_log = {
            "generation": generation,
            "best_weighted_fitness": float(best_weighted_fitness),
            "best_reader_score": float(best_reader_score),
            "best_retrieval_score": float(best_retrieval_score),
            "best_individual_text": best_individual_text,
            "success_achieved": bool(best_reader_score < self.success_threshold and 
                                   best_retrieval_score < self.success_threshold),
            "population_stats": population_stats,
            "timestamp": datetime.now().isoformat()
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
    
    def calculate_population_stats(self, fitness_weighted, score1, score2):

        return {
            "fitness_stats": {
                "mean": float(np.mean(fitness_weighted)),
                "std": float(np.std(fitness_weighted)),
                "min": float(np.min(fitness_weighted)),
                "max": float(np.max(fitness_weighted)),
                "median": float(np.median(fitness_weighted))
            },
            "reader_score_stats": {
                "mean": float(np.mean(score1)),
                "std": float(np.std(score1)),
                "min": float(np.min(score1)),
                "max": float(np.max(score1)),
                "median": float(np.median(score1))
            },
            "retrieval_score_stats": {
                "mean": float(np.mean(score2)),
                "std": float(np.std(score2)),
                "min": float(np.min(score2)),
                "max": float(np.max(score2)),
                "median": float(np.median(score2))
            }
        }

    def save_logs(self):
        log_data = {
            "experiment_info": {
                "question": self.question,
                "answer": self.answer,
                "n_iterations": self.n_iter,
                "population_size": self.pop.pop_size,
                "tournament_size": self.tournament_size,
                "success_threshold": self.success_threshold,
                "start_time": self.generation_logs[0]["timestamp"] if self.generation_logs else None,
                "end_time": self.generation_logs[-1]["timestamp"] if self.generation_logs else None
            },
            "final_results": {
                "success_achieved": self.success_achieved,
                "success_generation": self.success_generation,
                "best_fitness": float(self.best_fitness) if self.best_fitness is not None else None,
                "best_reader_score": float(self.best_score1) if self.best_score1 is not None else None,
                "best_retrieval_score": float(self.best_score2) if self.best_score2 is not None else None,
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
        P_fitness_weighted, P_score1, P_score2 = self.fitness(
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

            O_fitness_weighted, O_score1, O_score2 = self.fitness(
                question=self.question,
                contexts=[ind.get_perturbed_text() for ind in O],
                answer=self.answer
            )

            # pool
            pool = P + O
            pool_fitness_weighted = np.concatenate([P_fitness_weighted, O_fitness_weighted], axis=0)
            pool_score1 = np.concatenate([P_score1, O_score1], axis=0)
            pool_score2 = np.concatenate([P_score2, O_score2], axis=0)

            # Find best individual in current generation (LOWEST weighted fitness = BEST)
            current_best_idx = np.argmin(pool_fitness_weighted)
            current_best_fitness = pool_fitness_weighted[current_best_idx]
            current_best_individual = pool[current_best_idx]
            current_best_score1 = pool_score1[current_best_idx]  # reader_score
            current_best_score2 = pool_score2[current_best_idx]  # retrieval_score
            
            # Update global best
            if self.best_fitness is None or current_best_fitness < self.best_fitness:
                self.best_fitness = current_best_fitness
                self.best_individual = current_best_individual
                self.best_score1 = current_best_score1
                self.best_score2 = current_best_score2
            

            pop_stats = self.calculate_population_stats(pool_fitness_weighted, pool_score1, pool_score2)
            
            # Log generation data
            self.log_generation(
                generation=iter_idx,
                best_weighted_fitness=current_best_fitness,
                best_reader_score=current_best_score2,
                best_retrieval_score=current_best_score1,
                best_individual_text=current_best_individual.get_perturbed_text(),
                population_stats=pop_stats
            )

            # Print generation info
            print(f"\n📊 Generation {iter_idx}")
            print(f"   Best weighted fitness: {current_best_fitness:.6f}")
            print(f"   Best reader score: {current_best_score2:.6f}")
            print(f"   Best retrieval score: {current_best_score1:.6f}")
            print(f"   Success criteria met: {current_best_score1 < self.success_threshold and current_best_score2 < self.success_threshold}")
            
            # Optional: print generated answer for debugging
            try:
                generated_answer = self.fitness.reader.generate(self.question, [current_best_individual.get_perturbed_text()])
                if isinstance(generated_answer, list):
                    generated_answer = generated_answer[0] if generated_answer else "No answer"
                    self.adv_output = generated_answer
                print(f"   Generated answer: '{generated_answer.strip()}'")
            except Exception as e:
                print(f"   Generated answer: Error - {str(e)}")
            
            # Early stopping if success achieved
            # if (self.success_achieved and 
            #     iter_idx - self.success_generation >= 10):  # Continue for 10 more generations after success
            #     print(f"🎯 Early stopping: Success maintained for 10 generations")
            #     break
            
            # Selection for next generation
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

        # Update population
        self.pop.individuals = P
        
        # Final results
        print("\n" + "="*60)
        print("🏁 GA Evolution Completed")
        print(f"   Final success status: {self.success_achieved}")
        if self.success_achieved:
            print(f"   Success achieved at generation: {self.success_generation}")
        print(f"   Best weighted fitness: {self.best_fitness:.6f}")
        print(f"   Best reader score: {self.best_score1:.6f}")
        print(f"   Best retrieval score: {self.best_score2:.6f}")
        print("="*60)
        
        # Save logs
        self.save_logs()
        
        return {
            "success": self.success_achieved,
            "success_generation": self.success_generation,
            "best_fitness": self.best_fitness,
            "best_reader_score": self.best_score1,
            "best_retrieval_score": self.best_score2,
            "best_individual": self.best_individual,
            "total_generations": iter_idx + 1,
            "log_file": self.log_file
        }

    def get_best_result(self):
        """
        Trả về kết quả tốt nhất
        """
        return {
            "individual": self.best_individual,
            "fitness": self.best_fitness,
            "reader_score": self.best_score1,
            "retrieval_score": self.best_score2,
            "text": self.best_individual.get_perturbed_text() if self.best_individual else None,
            "success": self.success_achieved
        }

    def print_summary(self):
        """
        In tóm tắt kết quả
        """
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