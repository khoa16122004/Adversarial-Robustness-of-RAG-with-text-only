
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
        return top1_d, question, gt_answer
    
    def len(self):
        return len(self.data)
    
from textattack.shared import AttackedText

def split(text):
    return AttackedText(text).words


def find_anwser(context_split, answer_split):
    results = []
    for i in range(len(context_split)):
        if context_split[i] in answer_split:
            results.append(i)
    return results


