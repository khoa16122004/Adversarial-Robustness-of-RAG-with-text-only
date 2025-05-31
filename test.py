from utils import split
from utils import set_seed_everything, DataLoader
import json
dataset = DataLoader("sample5_data.json")

for i in range(dataset.len()):
    original_text, question, gt_answer = dataset.take_sample(i)
    split_ = split(original_text)
    print("Split: ", split_)
    print("Answer: ", gt_answer)
    position = int(input())
    print(split_[position])