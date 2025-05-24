import argparse
from algorithm import GA
from population import create_population, Population, Individual
from fitness import WeightedSUm
import numpy as np
from utils import set_seed_everything


def main(args):
    
    set_seed_everything(22520691)
    
    question = "What fields has Khoa contributed to in his research career?"
    original_text = "Khoa developed a strong passion for artificial intelligence during his university years. After graduating with honors, he decided to pursue a career in research. In 2025, Khoa officially became a researcher at a leading technology institute. Since then, he has contributed to several groundbreaking projects in computer vision and natural language processing."

    answer = "Computer vision and NLP"
    
    
    
    if args.attack != "ga":
        raise ValueError(f"Unsupported attack method: {args.attack}")

    fitness = WeightedSUm(
        reader_name=args.reader_name,
        q_name=args.q_name,
        c_name=args.c_name,
        retriever_weight=args.retriever_weight,
        reader_weight=args.reader_weight,
        question=question,
        original_text=original_text,
        answer=answer
        
    )

    population = create_population(original_text, args)

    ga = GA(
        n_iter=args.n_iter,
        population=population,
        fitness=fitness,
        tournament_size=args.tournament_size,
        question=question,
        answer=answer
    )

    result = ga.solve_rule()
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
    parser.add_argument("--attack", type=str, default="ga", help="Attack method")
    parser.add_argument("--reader_name", default="Llama-7b", type=str, help="Retriever model name")
    parser.add_argument("--q_name", default="facebook/dpr-question_encoder-multiset-base", type=str, help="Question encoder name")
    parser.add_argument("--c_name", default="facebook/dpr-ctx_encoder-multiset-base", type=str, help="Context encoder name")
    parser.add_argument("--retriever_weight", type=float, default=0.5, help="Weight for retriever")
    parser.add_argument("--reader_weight", type=float, default=0.5, help="Weight for reader")
    parser.add_argument("--n_iter", type=int, default=10, help="Number of GA iterations")
    parser.add_argument("--tournament_size", type=int, default=4, help="Tournament size")
    parser.add_argument("--pop_size", type=int, default=4, help="Population size")
    
    parser.add_argument("--pct_words_to_swap", type=float, default=0.3, help="Percentage of words to swap")
    args = parser.parse_args()
    pop = main(args)
