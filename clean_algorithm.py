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

class BaseGA:
    """Base class for Genetic Algorithm implementations"""
    
    def __init__(self, n_iter, population, fitness, question, answer, 
                 log_dir="ga_logs", success_threshold=1.0, algorithm_name="GA"):
        self.n_iter = n_iter
        self.pop = population
        self.fitness = fitness
        self.question = question
        self.answer = answer
        self.success_threshold = success_threshold
        self.algorithm_name = algorithm_name
        
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.generation_logs = []
        self.best_individual = None
        self.best_fitness = None
        self.best_score1 = None  
        self.best_score2 = None  
        self.success_achieved = False
        self.success_generation = None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"{algorithm_name.lower()}_log_{timestamp}.json")

    def log_generation(self, generation, best_weighted_fitness, best_reader_score, best_retrieval_score, 
                      best_individual_text, population_stats, **kwargs):
        """Log generation data"""
        generation_log = {
            "generation": generation,
            "best_weighted_fitness": float(best_weighted_fitness),
            "best_reader_score": float(best_reader_score),
            "best_retrieval_score": float(best_retrieval_score),
            "best_individual_text": best_individual_text,
            "success_achieved": bool(best_reader_score < self.success_threshold and 
                                   best_retrieval_score < self.success_threshold),
            "population_stats": population_stats,
            "timestamp": datetime.now().isoformat(),
            **kwargs  # Additional stats from subclasses
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
        """Calculate population statistics"""
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
        """Save experiment logs to JSON file"""
        log_data = {
            "experiment_info": {
                "algorithm": self.algorithm_name,
                "question": self.question,
                "answer": self.answer,
                "n_iterations": self.n_iter,
                "population_size": self.pop.pop_size,
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
                "best_individual_text": self.best_individual.get_perturbed_text() if self.best_individual else None
            },
            "generation_logs": self.generation_logs
        }
        
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        
        print(f"📝 Logs saved to: {self.log_file}")

    def generate_offspring(self, P):
        """Generate offspring population through crossover and mutation"""
        O = []
        for _ in range(self.pop.pop_size // 2):
            parent_idx1, parent_idx2 = random.sample(range(self.pop.pop_size), 2)
            parent1, parent2 = P[parent_idx1], P[parent_idx2]
            offspring1, offspring2 = self.pop.crossover(parent1, parent2)
            offspring1 = self.pop.mutation(offspring1)
            offspring2 = self.pop.mutation(offspring2)
            O.extend([offspring1, offspring2])
        return O

    def update_best_individual(self, pool, pool_fitness_weighted, pool_score1, pool_score2):
        """Update best individual based on weighted fitness"""
        current_best_idx = np.argmin(pool_fitness_weighted)
        current_best_fitness = pool_fitness_weighted[current_best_idx]
        current_best_individual = pool[current_best_idx]
        current_best_score1 = pool_score1[current_best_idx]
        current_best_score2 = pool_score2[current_best_idx]
        
        if self.best_fitness is None or current_best_fitness < self.best_fitness:
            self.best_fitness = current_best_fitness
            self.best_individual = current_best_individual
            self.best_score1 = current_best_score1
            self.best_score2 = current_best_score2
        
        return current_best_fitness, current_best_individual, current_best_score1, current_best_score2

    def print_generation_info(self, iter_idx, current_best_fitness, current_best_score1, 
                            current_best_score2, current_best_individual, **kwargs):
        """Print generation information"""
        print(f"\n📊 Generation {iter_idx}")
        print(f"   Best weighted fitness: {current_best_fitness:.6f}")
        print(f"   Best reader score: {current_best_score2:.6f}")
        print(f"   Best retrieval score: {current_best_score1:.6f}")
        
        # Print additional info from subclasses
        for key, value in kwargs.items():
            print(f"   {key}: {value}")
        
        print(f"   Success criteria met: {current_best_score1 < self.success_threshold and current_best_score2 < self.success_threshold}")
        
        # Print generated answer for debugging
        try:
            generated_answer = self.fitness.reader.generate(self.question, [current_best_individual.get_perturbed_text()])
            if isinstance(generated_answer, list):
                generated_answer = generated_answer[0] if generated_answer else "No answer"
            print(f"   Generated answer: '{generated_answer.strip()}'")
        except Exception as e:
            print(f"   Generated answer: Error - {str(e)}")

    def check_early_stopping(self, iter_idx):
        """Check if early stopping criteria is met"""
        return (self.success_achieved and 
                iter_idx - self.success_generation >= 10)

    def selection(self, pool_indices, pool_fitness_weighted, pool_score1, pool_score2):
        """Selection method - to be implemented by subclasses"""
        raise NotImplementedError("Selection method must be implemented by subclasses")

    def solve_rule(self):
        """Main evolution loop"""
        P = self.pop.individuals
        P_fitness_weighted, P_score1, P_score2 = self.fitness(
            question=self.question,
            contexts=[ind.get_perturbed_text() for ind in P],
            answer=self.answer
        )

        print(f"🚀 Starting {self.algorithm_name} Evolution")
        print(f"   Population size: {self.pop.pop_size}")
        print(f"   Generations: {self.n_iter}")
        print(f"   Success threshold: {self.success_threshold}")
        print("="*60)

        for iter_idx in tqdm(range(self.n_iter), desc=f"{self.algorithm_name} Evolution"):
            # Generate offspring
            O = self.generate_offspring(P)

            # Evaluate offspring
            O_fitness_weighted, O_score1, O_score2 = self.fitness(
                question=self.question,
                contexts=[ind.get_perturbed_text() for ind in O],
                answer=self.answer
            )

            # Create combined pool
            pool = P + O
            pool_fitness_weighted = np.concatenate([P_fitness_weighted, O_fitness_weighted], axis=0)
            pool_score1 = np.concatenate([P_score1, O_score1], axis=0)
            pool_score2 = np.concatenate([P_score2, O_score2], axis=0)

            # Update best individual
            current_best_fitness, current_best_individual, current_best_score1, current_best_score2 = \
                self.update_best_individual(pool, pool_fitness_weighted, pool_score1, pool_score2)

            # Calculate statistics
            pop_stats = self.calculate_population_stats(pool_fitness_weighted, pool_score1, pool_score2)
            
            # Get additional stats from subclasses
            additional_stats = self.get_additional_stats(pool_score1, pool_score2)
            
            # Log generation data
            self.log_generation(
                generation=iter_idx,
                best_weighted_fitness=current_best_fitness,
                best_reader_score=current_best_score2,
                best_retrieval_score=current_best_score1,
                best_individual_text=current_best_individual.get_perturbed_text(),
                population_stats=pop_stats,
                **additional_stats
            )

            # Print generation info
            generation_info = self.get_generation_info()
            self.print_generation_info(iter_idx, current_best_fitness, current_best_score1, 
                                     current_best_score2, current_best_individual, **generation_info)
            
            # Early stopping check
            if self.check_early_stopping(iter_idx):
                print(f"🎯 Early stopping: Success maintained for 10 generations")
                break
            
            # Selection for next generation
            pool_indices = np.arange(len(pool))
            selected_indices = self.selection(pool_indices, pool_fitness_weighted, pool_score1, pool_score2)
            
            # Update population
            P = [pool[i] for i in selected_indices]
            P_fitness_weighted = pool_fitness_weighted[selected_indices]
            P_score1 = pool_score1[selected_indices]
            P_score2 = pool_score2[selected_indices]

        # Update population and finalize
        self.pop.individuals = P
        self.finalize_evolution(iter_idx)
        
        return self.get_results(iter_idx)

    def get_additional_stats(self, pool_score1, pool_score2):
        """Get additional statistics for logging - to be overridden by subclasses"""
        return {}

    def get_generation_info(self):
        """Get additional generation info for printing - to be overridden by subclasses"""
        return {}

    def finalize_evolution(self, iter_idx):
        """Finalize evolution process"""
        print("\n" + "="*60)
        print(f"🏁 {self.algorithm_name} Evolution Completed")
        print(f"   Final success status: {self.success_achieved}")
        if self.success_achieved:
            print(f"   Success achieved at generation: {self.success_generation}")
        print(f"   Best weighted fitness: {self.best_fitness:.6f}")
        print(f"   Best reader score: {self.best_score1:.6f}")
        print(f"   Best retrieval score: {self.best_score2:.6f}")
        print("="*60)
        
        self.save_logs()

    def get_results(self, iter_idx):
        """Get final results"""
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
        """Return best result"""
        return {
            "individual": self.best_individual,
            "fitness": self.best_fitness,
            "reader_score": self.best_score1,
            "retrieval_score": self.best_score2,
            "text": self.best_individual.get_perturbed_text() if self.best_individual else None,
            "success": self.success_achieved
        }

    def print_summary(self):
        """Print evolution summary"""
        print("\n" + "="*60)
        print(f"📋 {self.algorithm_name} EVOLUTION SUMMARY")
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
            best_text = self.best_individual.get_perturbed_text()
            print(f"   Text: {best_text[:200]}{'...' if len(best_text) > 200 else ''}")
            
            try:
                generated = self.fitness.reader.generate(self.question, [self.best_individual.get_perturbed_text()])
                if isinstance(generated, list):
                    generated = generated[0] if generated else "No answer"
                print(f"   Generated Answer: '{generated.strip()}'")
            except Exception as e:
                print(f"   Generated Answer: Error - {str(e)}")
        
        # Print algorithm-specific summary
        self.print_algorithm_specific_summary()
        print("="*60)

    def print_algorithm_specific_summary(self):
        """Print algorithm-specific summary - to be overridden by subclasses"""
        pass


class GA(BaseGA):
    """Traditional Genetic Algorithm with tournament selection"""
    
    def __init__(self, n_iter, population, fitness, tournament_size, question, answer, 
                 log_dir="ga_logs", success_threshold=1.0):
        super().__init__(n_iter, population, fitness, question, answer, 
                        log_dir, success_threshold, "GA")
        self.tournament_size = tournament_size

    def tournament_selection(self, fitness_array: np.ndarray, tournament_size: int = 4):
        """Tournament selection implementation"""
        selected_indices = []
        for j in range(0, len(fitness_array), tournament_size):
            group_fitness = fitness_array[j : j + tournament_size]
            best_in_group = j + np.argmin(group_fitness)
            selected_indices.append(best_in_group)
        return selected_indices

    def selection(self, pool_indices, pool_fitness_weighted, pool_score1, pool_score2):
        """Tournament selection for next generation"""
        selected_pool_index_parts = []
        for _ in range(self.tournament_size // 2):
            np.random.shuffle(pool_indices)
            selected = self.tournament_selection(pool_fitness_weighted[pool_indices])
            selected_pool_index_parts.append(pool_indices[selected])
        selected_pool_index = np.concatenate(selected_pool_index_parts)
        return selected_pool_index

    def save_logs(self):
        """Override to add tournament_size to experiment info"""
        log_data = {
            "experiment_info": {
                "algorithm": self.algorithm_name,
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
                "best_individual_text": self.best_individual.get_perturbed_text() if self.best_individual else None
            },
            "generation_logs": self.generation_logs
        }
        
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        
        print(f"📝 Logs saved to: {self.log_file}")


class NSGAII(BaseGA):
    """NSGA-II Multi-objective Genetic Algorithm"""
    
    def __init__(self, n_iter, population, fitness, question, answer, 
                 log_dir="nsgaii_logs", success_threshold=1.0):
        super().__init__(n_iter, population, fitness, question, answer, 
                        log_dir, success_threshold, "NSGA-II")
        self.nds = NonDominatedSorting()

    def calculate_crowding_distance(self, objectives):
        """Calculate crowding distance for individuals based on their objectives"""
        n_points, n_obj = objectives.shape
        
        if n_points <= 2:
            return np.full(n_points, np.inf)
        
        crowding_distance = np.zeros(n_points)
        
        for m in range(n_obj):
            sorted_indices = np.argsort(objectives[:, m])
            crowding_distance[sorted_indices[0]] = np.inf
            crowding_distance[sorted_indices[-1]] = np.inf
            
            obj_range = objectives[sorted_indices[-1], m] - objectives[sorted_indices[0], m]
            
            if obj_range > 0:
                for i in range(1, n_points - 1):
                    crowding_distance[sorted_indices[i]] += (
                        objectives[sorted_indices[i + 1], m] - 
                        objectives[sorted_indices[i - 1], m]
                    ) / obj_range
        
        return crowding_distance

    def selection(self, pool_indices, pool_fitness_weighted, pool_score1, pool_score2):
        """NSGA-II selection based on non-dominated sorting and crowding distance"""
        objectives = np.column_stack([pool_score1, pool_score2])
        fronts = self.nds.do(objectives, only_non_dominated_front=False)
        
        selected_indices = []
        
        for front in fronts:
            if len(selected_indices) + len(front) <= self.pop.pop_size:
                selected_indices.extend(pool_indices[front])
            else:
                front_objectives = objectives[front]
                crowding_dist = self.calculate_crowding_distance(front_objectives)
                sorted_crowding_indices = np.argsort(-crowding_dist)
                remaining = self.pop.pop_size - len(selected_indices)
                selected_from_front = front[sorted_crowding_indices[:remaining]]
                selected_indices.extend(pool_indices[selected_from_front])
                break
        
        return selected_indices

    def calculate_pareto_stats(self, objectives):
        """Calculate Pareto front statistics"""
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

    def get_additional_stats(self, pool_score1, pool_score2):
        """Get Pareto front statistics for logging"""
        objectives = np.column_stack([pool_score1, pool_score2])
        return {"pareto_stats": self.calculate_pareto_stats(objectives)}

    def get_generation_info(self):
        """Get Pareto front info for printing"""
        if hasattr(self, '_current_pareto_stats'):
            return {
                "Pareto front size": self._current_pareto_stats['pareto_front_size'],
                "Total fronts": self._current_pareto_stats['total_fronts']
            }
        return {}

    def print_generation_info(self, iter_idx, current_best_fitness, current_best_score1, 
                            current_best_score2, current_best_individual, **kwargs):
        """Store pareto stats for generation info"""
        if 'pareto_stats' in kwargs:
            self._current_pareto_stats = kwargs['pareto_stats']
        super().print_generation_info(iter_idx, current_best_fitness, current_best_score1, 
                                    current_best_score2, current_best_individual, 
                                    **self.get_generation_info())

    def get_pareto_front(self):
        """Get current Pareto front individuals"""
        if not self.pop.individuals:
            return []
        
        fitness_weighted, score1, score2 = self.fitness(
            question=self.question,
            contexts=[ind.get_perturbed_text() for ind in self.pop.individuals],
            answer=self.answer
        )
        
        objectives = np.column_stack([score1, score2])
        fronts = self.nds.do(objectives, only_non_dominated_front=True)
        pareto_indices = fronts[0]
        
        pareto_individuals = []
        for idx in pareto_indices:
            pareto_individuals.append({
                "individual": self.pop.individuals[idx],
                "reader_score": score2[idx],
                "retrieval_score": score1[idx],
                "weighted_fitness": fitness_weighted[idx],
                "text": self.pop.individuals[idx].get_perturbed_text()
            })
        
        return pareto_individuals

    def print_algorithm_specific_summary(self):
        """Print NSGA-II specific summary"""
        try:
            pareto_front = self.get_pareto_front()
            print(f"\n🔄 PARETO FRONT:")
            print(f"   Size: {len(pareto_front)}")
            if pareto_front:
                reader_scores = [p['reader_score'] for p in pareto_front]
                retrieval_scores = [p['retrieval_score'] for p in pareto_front]
                print(f"   Reader Score Range: [{min(reader_scores):.6f}, {max(reader_scores):.6f}]")
                print(f"   Retrieval Score Range: [{min(retrieval_scores):.6f}, {max(retrieval_scores):.6f}]")
        except Exception as e:
            print(f"   Pareto Front: Error calculating - {str(e)}")