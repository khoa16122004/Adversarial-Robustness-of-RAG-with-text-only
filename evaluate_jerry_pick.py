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

    for i in range(len_dataset):
        original_text, question, gt_answer, answer_position_indices = dataset.take_sample(i)
        golden_answer = reader.generate(question, [original_text])[0]
        print("Golden answer: ", golden_answer)
        print("Gt answer: ", gt_answer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GA attack")
    parser.add_argument("--reader_name", default="Llama-7b", type=str, help="Retriever model name")
    parser.add_argument("--q_name", default="facebook/dpr-question_encoder-multiset-base", type=str, help="Question encoder name")
    parser.add_argument("--c_name", default="facebook/dpr-ctx_encoder-multiset-base", type=str, help="Context encoder name")
    parser.add_argument("--data_path", default="sample5_data.json", type=str, help="Data path")
    args = parser.parse_args()
    main(args)