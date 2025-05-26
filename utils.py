
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
    
def split(text):
    split_text = re.findall(r'\b\w+\b', text.lower())
    return split_text

def find_anser(context, anser):
    context_split = split(context)
    anser_split = split(anser)
    results = []
    for i in range(len(context_split)):
        if context_split[i] in anser_split:
            results.append(i)
    return results

