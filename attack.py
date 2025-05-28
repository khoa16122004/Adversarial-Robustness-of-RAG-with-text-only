import argparse
from algorithm import GA, NSGAII
from population import create_population, Population, Individual
from fitness import WeightedSUm, MultiScore, Targeted_MultiScore, Targeted_WeightedSUm
import numpy as np
from utils import set_seed_everything
from typo_transformation import ComboTypoTransformation

def main(args):
    
    set_seed_everything(22520691)
    
    question = "What is the fastest land animal?"
    original_text = "The cheetah is the fastest land animal, capable of reaching speeds up to 70 mph. It has a slender build and distinctive spotted coat. Cheetahs primarily hunt gazelles and other small antelopes in Africa."

    answer = "Cheetah"
    
  
    
    population = Population(
        original_text=original_text,
        answer=answer,
        pop_size=args.pop_size,
    )
    



    if args.fitness_statery == "golden_answer":
        if args.algorithm == "GA":
            fitness = WeightedSUm(
                reader_name=args.reader_name,
                q_name=args.q_name,
                c_name=args.c_name,
                retriever_weight=args.retriever_weight,
                reader_weight=args.reader_weight,
                question=question,
                original_text=original_text,
                answer=answer,
                
            )
            
            algo = GA(
                n_iter=args.n_iter,
                population=population,
                fitness=fitness,
                tournament_size=args.tournament_size,
                question=question,
                answer=answer
            )
            
            
            

            
        elif args.algorithm == "NSGAII":
            fitness = MultiScore(
                reader_name=args.reader_name,
                q_name=args.q_name,
                c_name=args.c_name,
                retriever_weight=args.retriever_weight,
                reader_weight=args.reader_weight,
                question=question,
                original_text=original_text,
                answer=answer
            )
            algo = NSGAII(
                n_iter=args.n_iter,
                population=population,
                fitness=fitness,
                question=question,
                answer=answer
            )
    
    
    elif args.fitness_statery == "target_answer":
        if args.algorithm == "GA":
            fitness = Targeted_WeightedSUm(
                reader_name=args.reader_name,
                q_name=args.q_name,
                c_name=args.c_name,
                retriever_weight=args.retriever_weight,
                reader_weight=args.reader_weight,
                question=question,
                original_text=original_text,
                answer=answer,
                target_text=args.target
            )
            
            algo = GA(
                n_iter=args.n_iter,
                population=population,
                fitness=fitness,
                tournament_size=args.tournament_size,
                question=question,
                answer=answer
            )
            
            
            

            
        elif args.algorithm == "NSGAII":
            fitness = Targeted_MultiScore(
                reader_name=args.reader_name,
                q_name=args.q_name,
                c_name=args.c_name,
                retriever_weight=args.retriever_weight,
                reader_weight=args.reader_weight,
                question=question,
                original_text=original_text,
                answer=answer
            )
            
            algo = NSGAII(
                n_iter=args.n_iter,
                population=population,
                fitness=fitness,
                question=question,
                answer=answer
            )

    result = algo.solve_rule()
    print("Best fitness: ", result["best_fitness"])
    print("Best best_reader_score: ", result["best_reader_score"])
    print("Best best_retrieval_score: ", result["best_retrieval_score"])
    print("Best best_individual_text: ", result["best_individual"])
    
    adv_output = fitness.reader.generate(question, [result["best_individual"].get_perturbed_text()])[0]
    print("Adv_output: ", adv_output)

    original_output = fitness.reader.generate(question, [original_text])[0]
    print("Original output: ", original_output)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GA attack")
    parser.add_argument("--reader_name", default="Llama-7b", type=str, help="Retriever model name")
    parser.add_argument("--q_name", default="facebook/dpr-question_encoder-multiset-base", type=str, help="Question encoder name")
    parser.add_argument("--c_name", default="facebook/dpr-ctx_encoder-multiset-base", type=str, help="Context encoder name")
    parser.add_argument("--retriever_weight", type=float, default=0.5, help="Weight for retriever")
    parser.add_argument("--reader_weight", type=float, default=0.5, help="Weight for reader")
    parser.add_argument("--n_iter", type=int, default=10, help="Number of GA iterations")
    parser.add_argument("--tournament_size", type=int, default=4, help="Tournament size")
    parser.add_argument("--pop_size", type=int, default=4, help="Population size")
    parser.add_argument("--fitness_statery", type=str, default="golden_answer", choices=['golden_answer', 'target_answer'], help="Fitness strategy")
    parser.add_argument("--algorithm", type=str, default="GA", choices=['GA', 'NSGAII'], help="Algorithm")
    parser.add_argument("--pct_words_to_swap", type=float, default=0.3, help="Percentage of words to swap")
    parser.add_argument("--target_answer", type=str, default="don't know", help="Target answer")
    args = parser.parse_args()
    pop = main(args)
