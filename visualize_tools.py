import json
import matplotlib.pyplot as plt

def generation_comparison_plot(ga_retri, ga_reader, nsga_retri, nsga_reader, title=None):
    generations_ga = list(range(len(ga_retri)))
    generations_nsga = list(range(len(nsga_retri)))

    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(title if title else "Comparison of GA vs NSGA-II", fontsize=16)

    axs[0].plot(generations_ga, ga_retri, marker='o', color='blue', label='GA')
    axs[0].plot(generations_nsga, nsga_retri, marker='s', color='orange', label='NSGA-II')
    axs[0].set_ylabel("Retrieval Score")
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(generations_ga, ga_reader, marker='o', color='green', label='GA')
    axs[1].plot(generations_nsga, nsga_reader, marker='s', color='red', label='NSGA-II')
    axs[1].set_ylabel("Reader Score")
    axs[1].set_xlabel("Generation")
    axs[1].grid(True)
    axs[1].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()



import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

def create_nsga_gif(nsga_retri_scores, nsga_reader_scores, filename="nsga_animation.gif", interval=1000):
    fig, ax = plt.subplots(figsize=(7,5))
    scat = ax.scatter([], [], alpha=0.6, color='blue')
    ax.set_xlim(min(min(nsga_retri_scores, key=min)), max(max(nsga_retri_scores, key=max)))
    ax.set_ylim(min(min(nsga_reader_scores, key=min)), max(max(nsga_reader_scores, key=max)))
    ax.set_xlabel("Retrieval Score")
    ax.set_ylabel("Reader Score")
    ax.grid(True)
    
    def update(frame):
        x = nsga_retri_scores[frame]
        y = nsga_reader_scores[frame]
        ax.set_title(f"NSGA-II Generations - Generation {frame}")
        scat.set_offsets(list(zip(x, y)))
        return scat,
    
    anim = FuncAnimation(fig, update, frames=len(nsga_retri_scores), interval=interval, blit=True)
    anim.save(filename, writer=PillowWriter(fps=1000//interval))
    plt.close(fig)

if __name__ == "__main__":
    # path_ga = r"D:\EnhanceGARAG\ga_logs\ga_golden_answer_0.json"
    # path_nsga = r"D:\EnhanceGARAG\nsgaii_logs\NSGAII_golden_answer_0.2_0.json"

    # with open(path_ga, 'r') as f:
    #     data_ga = json.load(f)['generation_logs']
    #     ga_retri_scores = [item['best_retrieval_score'] for item in data_ga]
    #     ga_reader_scores = [item['best_reader_score'] for item in data_ga]

    # with open(path_nsga, 'r') as f:
    #     data_nsga = json.load(f)['generation_logs']
    #     nsga_retri_scores = [item['best_retrieval_score'] for item in data_nsga]
    #     nsga_reader_scores = [item['best_reader_score'] for item in data_nsga]

    # generation_comparison_plot(
    #     ga_retri_scores, ga_reader_scores,
    #     nsga_retri_scores, nsga_reader_scores,
    #     title="GA vs NSGA-II Generation Score Comparison"
    # )
    
    path_NSGAII = r"D:\EnhanceGARAG\nsgaii_logs\NSGAII_golden_answer_0.2_0.json"
    with open(path_NSGAII, 'r') as f:
        data_nsga = json.load(f)['generation_logs']
        nsga_retri_scores = [item['all_scores']['retrieval_scores'] for item in data_nsga]
        nsga_reader_scores = [item['all_scores']['reader_scores'] for item in data_nsga]

    animate_nsga_generations(nsga_retri_scores, nsga_reader_scores, title="NSGA-II Generations Animation")