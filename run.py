import argparse
from algorithm import GA, NSGAII
from population import create_population, Population, Individual
from fitness import WeightedSUm, MultiScore, Targeted_MultiScore, Targeted_WeightedSUm
import numpy as np
from utils import set_seed_everything, DataLoader
from reader import Reader

def main(args):
    
    set_seed_everything(22520691)
    
    dataset = DataLoader(args.data_path)
    len_dataset = dataset.len()
    print("Num sample: ", len_dataset)
    reader = Reader(args.reader_name)
    
    
    for i in range(len_dataset):
        original_text, question, gt_answer = dataset[i]
        golden_answer = reader.generate(question, [original_text])
        population = create_population(original_text, golden_answer, args)



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
                    answer=golden_answer,
                    sample_id=i,
                )
                
                algo = GA(
                    n_iter=args.n_iter,
                    population=population,
                    fitness=fitness,
                    tournament_size=args.tournament_size,
                    question=question,
                    answer=golden_answer
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
                    answer=golden_answer
                )
                algo = NSGAII(
                    sample_id=i,
                    n_iter=args.n_iter,
                    population=population,
                    fitness=fitness,
                    question=question,
                    answer=golden_answer
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
                    answer=golden_answer,
                    target_text=args.target
                )
                
                algo = GA(
                    sample_id=i,
                    n_iter=args.n_iter,
                    population=population,
                    fitness=fitness,
                    tournament_size=args.tournament_size,
                    question=question,
                    answer=golden_answer
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
                    answer=golden_answer
                )
                
                algo = NSGAII(
                    n_iter=args.n_iter,
                    population=population,
                    fitness=fitness,
                    question=question,
                    answer=golden_answer
                )

        algo.solve_rule()
        
        
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
