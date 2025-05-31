from utils import split
from utils import set_seed_everything, DataLoader
import json
dataset = DataLoader("sample5_data.json")

for i in range(len(dataset)):
    original_text, question, gt_answer = dataset.take_sample(i)
    split_ = split(original_text)
    position = int(input())
    print(split_[position])