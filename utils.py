
import os
import random
import numpy as np
import torch
import json
import re
import pickle as pkl
from tqdm import tqdm
import matplotlib.animation as animation
import matplotlib.pyplot as plt

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
        # print("Data id: ", data['sample_id'])
        # print("Datakey: ", data.key())
        top1_d = data['documents'][0]
        question = data['question']
        gt_answer = data['answer']
        answer_position_indices = data['keyword_idx']
        # answer_position_indices = None
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

def dominate(a, b):
    if a[0] < b[0] and a[1] < b[1]:
        return 1
    elif b[0] < a[0] and b[1] < a[1]:
        return -1
    else:
        return 0

def arkiv_proccess(history):
    arkiv = history[0][:]
    final_history = [arkiv[:]]  

    for i in range(1, len(history)):
        current_gen = history[i]
        valid_new_idx = []
        remove_arkv_idx = []

        for j in range(len(current_gen)):
            is_valid = True
            for k in range(len(arkiv)):
                check = dominate(arkiv[k], current_gen[j])
                if check == 1:
                    is_valid = False
                    break
                elif check == -1:
                    remove_arkv_idx.append(k)
            if is_valid:
                valid_new_idx.append(j)

        remove_arkv_idx = list(set(remove_arkv_idx))
        arkiv = [ind for idx, ind in enumerate(arkiv) if idx not in remove_arkv_idx]

        for j in valid_new_idx:
            arkiv.append(current_gen[j])

        final_history.append(arkiv[:])  
    return final_history

def greedy_selection(final_font):
    scores = final_font[:, :2] # n x 2
    mask = scores[:, 0] < 1
    success = False

    if np.any(mask):
        selected_idx = np.argmin(scores[mask][:, 1])
        final_indices = np.where(mask)[0]
        if scores[mask][selected_idx, 0] < 1:
            success = True
        return final_font[final_indices[selected_idx]], success
    else:
        selected_idx = np.argmin(scores[:, 0])
        return final_font[selected_idx], success


def arkiv_multiple_font(dir, model_name, sample_id):
    pcts = [0.05, 0.1, 0.2, 0.5]
    pcts_list = {
        pct: None for pct in pcts
    }
    merge_font = []
    pcts_font = []
    
    path_template = r"{model_name}_ngsgaii_golden_answer_{pct}_{id}.pkl"
    for pct in pcts: 
        path = path_template.format(model_name=model_name, 
                                    pct=pct, 
                                    id=sample_id)
        full_path = os.path.join(dir, path)
        
        history = pkl.load(open(full_path, "rb"))
        history = arkiv_proccess(history)
        
        # for each pct
        pct_final_font = history[-1]
        selected_ind, success = greedy_selection(np.array(pct_final_font))
        text = selected_ind[2].get_perturbed_text()
        pcts_list[pct] = text
        pcts_font.append(history)
        # merge font
        merge_font = [arkiv_proccess([pct_final_font] + merge_font)[-1]]
        
    
    selected_ind, success = greedy_selection(np.array(merge_font[-1]))
    text = selected_ind[2].get_perturbed_text()
    pcts_list['merge'] = text
    return pcts_list, pcts_font, merge_font

def visualize_process(final_history, objective_labels=["L_RSR", "L_GPR"], interval=500):
    num_generations = len(final_history)

    # Convert to numpy arrays if not already
    final_history = [np.array(gen) for gen in final_history]

    # Determine bounds across all generations
    min_obj1 = min(np.min(gen[:, 0]) for gen in final_history)
    max_obj1 = max(np.max(gen[:, 0]) for gen in final_history)
    min_obj2 = min(np.min(gen[:, 1]) for gen in final_history)
    max_obj2 = max(np.max(gen[:, 1]) for gen in final_history)

    padding_obj1 = (max_obj1 - min_obj1) * 0.1
    padding_obj2 = (max_obj2 - min_obj2) * 0.1
    xlim = (min_obj1 - padding_obj1, max_obj1 + padding_obj1)
    ylim = (min_obj2 - padding_obj2, max_obj2 + padding_obj2)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter([], [], c='red')
    ax.set_xlabel(objective_labels[0])
    ax.set_ylabel(objective_labels[1])
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid(True)

    def update(frame):
        data = final_history[frame]
        scatter.set_offsets(data)
        ax.set_title(f"Pareto Front - Generation {frame + 1}")
        return scatter,

    ani = animation.FuncAnimation(fig, update, frames=num_generations, interval=interval, blit=True)
    plt.close(fig)
    return ani

def visualize_process_multiple(final_histories, objective_labels=["L_RSR", "L_GPR"], interval=500):
    pcts = ["0.05", "0.1", "0.2", "0.5"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    
    num_generations = min(len(h) for h in final_histories)

    # Ensure all elements are numpy arrays
    # final_histories = [[np.array(gen) for gen in history] for history in final_histories]

    # Flatten all generations from all histories to compute global bounds
    all_points = np.vstack([gen for history in final_histories for gen in history])

    min_obj1, max_obj1 = np.min(all_points[:, 0]), np.max(all_points[:, 0])
    min_obj2, max_obj2 = np.min(all_points[:, 1]), np.max(all_points[:, 1])

    padding_obj1 = (max_obj1 - min_obj1) * 0.1
    padding_obj2 = (max_obj2 - min_obj2) * 0.1

    xlim = (min_obj1 - padding_obj1, max_obj1 + padding_obj1)
    ylim = (min_obj2 - padding_obj2, max_obj2 + padding_obj2)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatters = [ax.scatter([], [], c=color, label=label) for color, label in zip(colors, pcts)]

    ax.set_xlabel(objective_labels[0])
    ax.set_ylabel(objective_labels[1])
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.legend()
    ax.grid(True)

    def update(frame):
        for scatter, history in zip(scatters, final_histories):
            scatter.set_offsets(history[frame])
        ax.set_title(f"Pareto Front - Generation {frame + 1}")
        return scatters

    ani = animation.FuncAnimation(fig, update, frames=num_generations, interval=interval, blit=True)
    plt.close(fig)
    return ani

def mean_proccess(model_name, dir):
    path_template = r"{model_name}_ngsgaii_golden_answer_{pct}_{id}.pkl"
    final_proccess = []
    for id in range(10):
        proccess_full = []
        for i, pct in enumerate([0.05, 0.1, 0.2, 0.5]):
            path = os.path.join(dir, path_template.format(model_name=model_name, pct=pct, id=id))
            history = pkl.load(open(path, "rb"))
            final_font = arkiv_proccess(history)
            final_font = [np.array(font) for font in final_font]

            proccess_retri_list = [np.min(font[:, 0]) for font in final_font]
            proccess_reader_list = [np.min(font[:, 1]) for font in final_font]
            proccess_full.append(np.column_stack((proccess_retri_list, proccess_reader_list)))
        
        proccess_full = np.array(proccess_full)
        final_proccess.append(proccess_full)

    final_proccess = np.array(final_proccess)
    proccess_mean = np.mean(final_proccess, axis=0)
    return proccess_mean

def plot_scores(proccess_full):
    pcts = [0.05, 0.1, 0.2, 0.5]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    generations = np.arange(proccess_full.shape[1])
    
    plt.figure(figsize=(10, 5))
    for i, (pct, color) in enumerate(zip(pcts, colors)):
        plt.plot(generations, proccess_full[i, :, 0], label=f'pct={pct}', color=color)
    plt.xlabel('Generation')
    plt.ylabel('L_RSR')
    plt.title('L_RSR Score over Generations')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Vẽ biểu đồ cho score 1 (GPR score)
    plt.figure(figsize=(10, 5))
    for i, (pct, color) in enumerate(zip(pcts, colors)):
        plt.plot(generations, proccess_full[i, :, 1], label=f'pct={pct}', color=color)
    plt.xlabel('Generation')
    plt.ylabel('GPR score')
    plt.title('GPR Score over Generations')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def get_font(dir_, model_name, sample_id):
    pcts = [0.05, 0.1, 0.2, 0.5]
    pcts_list, pcts_font, merge_font = arkiv_multiple_font(dir=dir_, 
                                                           model_name=model_name,
                                                           sample_id=sample_id)
    return np.array(merge_font[0])


# dir_ = "result/llama_7b_nsgaii_logs"
# model_name = "llama-7b"
# sample_id = 0
# get_font(dir_, model_name, sample_id)




