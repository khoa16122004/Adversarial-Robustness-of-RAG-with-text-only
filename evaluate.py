import argparse
from algorithm import GA, NSGAII
from population import  Population, Individual
from fitness import WeightedSUm, MultiScore, Targeted_MultiScore, Targeted_WeightedSUm
import numpy as np
from utils import set_seed_everything, DataLoader
from reader import Reader
from utils import find_answer, split
from fitness import MultiScore
def main(args):
    set_seed_everything(22520691)
    dataset = DataLoader(args.data_path)
    len_dataset = dataset.len()
    

    
    reader = Reader(args.reader_name)
    with open(args.adv_text_path, "r") as f:
        adv_texts = [line.strip() for line in f.readlines()]
    

    
    original_text, question, gt_answer = dataset.take_sample(args.evaluate_id)
    original_text = "0 1 2 3 4 5 6 7 8 9 10"

    fitness = MultiScore(args.reader_name, 
                    args.q_name, 
                    args.c_name, 
                    question, original_text, 'Launch')
    inference_text = adv_texts + [original_text]
    output = reader.generate(question, inference_text)
    print("Output: ", output)
    print("Fitness: ", fitness(question, inference_text, 'Launch'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GA attack")
    parser.add_argument("--reader_name", default="Llama-7b", type=str, help="Retriever model name")
    parser.add_argument("--q_name", default="facebook/dpr-question_encoder-multiset-base", type=str, help="Question encoder name")
    parser.add_argument("--c_name", default="facebook/dpr-ctx_encoder-multiset-base", type=str, help="Context encoder name")
    parser.add_argument("--data_path", default="sample5_data.json", type=str, help="Data path")
    parser.add_argument("--evaluate_id", default=0, type=int, help="Evaluate id")
    parser.add_argument("--adv_text_path", type=str)
    args = parser.parse_args()
    main(args)