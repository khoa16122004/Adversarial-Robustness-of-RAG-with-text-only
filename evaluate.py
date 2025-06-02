import argparse
from algorithm import GA, NSGAII
from population import  Population, Individual
from fitness import WeightedSUm, MultiScore, Targeted_MultiScore, Targeted_WeightedSUm
import numpy as np
from utils import set_seed_everything, DataLoader
from reader import Reader
from utils import find_answer, split
from fitness import MultiScore
import os
def main(args):
    set_seed_everything(22520691)
    dataset = DataLoader(args.data_path)
    len_dataset = dataset.len()
    reader = Reader(args.reader_name)
    
    pcts = [0.05, 0.1, 0.2, 0.5]
    for pct in pcts:
        print("=======================================")
        print("PCT: ", pct)
        dir_ = f"pertubed_text/{args.reader_name}_{pct}"
        for i in range(len_dataset):
            path = os.path.join(dir_, f"{i}.txt")
            original_text, question, gt_answer, _ = dataset.take_sample(i)
            adv_text = open(path, "r").readline().strip()
            contexts = [original_text, adv_text]
            output = reader.generate(question, contexts)
            print("GT Answer: ", gt_answer)
            print("Output: ", output)
            
            fitness = MultiScore(
                reader_name=args.reader_name,
                q_name="facebook/dpr-question_encoder-multiset-base",
                c_name="facebook/dpr-ctx_encoder-multiset-base",
                question=question,
                original_text=original_text,
                answer=gt_answer
            )
            
            score = fitness(question, [adv_text], gt_answer)
            print("Score: ", score)
            
                        

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GA attack")
    parser.add_argument("--reader_name", default="Llama-7b", type=str, help="Retriever model name")
    parser.add_argument("--q_name", default="facebook/dpr-question_encoder-multiset-base", type=str, help="Question encoder name")
    parser.add_argument("--c_name", default="facebook/dpr-ctx_encoder-multiset-base", type=str, help="Context encoder name")
    parser.add_argument("--data_path", default="data_new_v2.json", type=str, help="Data path")
    args = parser.parse_args()
    main(args)