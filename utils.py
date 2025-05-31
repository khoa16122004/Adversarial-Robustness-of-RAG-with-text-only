
import os
import random
import numpy as np
import torch
import json
import re

def set_seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class DataLoader:
    def __init__(self, file_path):
        self.data = self.load(file_path)
        
    def load(self, file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
        return data

    def take_sample(self, index):
        data = self.data[index]
        top1_d = data['documents'][0]
        question = data['question']
        gt_answer = data['answer']
        answer_position_indices = data['answer_position']
        return top1_d, question, gt_answer, answer_position_indices
    
    def len(self):
        return len(self.data)
    
from textattack.shared import AttackedText

def split(text):
    return AttackedText(text).words


def find_answer(context_split, answer_split):
    results = []
    print("Context: ", context_split)
    print("Answer: ", answer_split)
    for i in range(len(context_split)):
        if context_split[i] in answer_split:
            results.append(i)
    return results


def exact_match(prediction: str, ground_truth: str) -> bool:
    return prediction == ground_truth

def accuracy_span_inclusion(prediction: str, ground_truth_span: str) -> bool:
    return ground_truth_span in prediction


