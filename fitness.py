from reader import Reader
from retrieval import Retriever
import numpy as np
from utils import set_seed_everything

set_seed_everything(222520691)

class FitnessReader:
    def __init__(self, reader_name, original_answer: str):
        self.reader = Reader(reader_name)

    def __call__(self, question, contexts, answer):
        # Gọi reader với args phù hợp
        return self.reader(question, contexts, answer)

class FitnessRetriever:
    def __init__(self, q_name, c_name):
        self.retriever = Retriever(q_name, c_name)

    def __call__(self, *args, **kwargs):
        # Gọi retriever với args phù hợp
        return self.retriever(*args, **kwargs)

class FitnessDual:
    def __init__(self, retriever_name, q_name, c_name):
        self.reader = Reader(retriever_name)
        self.retriever = Retriever(q_name, c_name)

    def __call__(self, question, contexts, answer):
        
        retrieval_result = self.retriever(question, contexts)
        reader_result = self.reader(question, contexts, answer)
        stacked_result = np.stack([retrieval_result, reader_result], axis=1)

        return stacked_result
    
    
class WeightedSUm:
    def __init__(self, reader_name, q_name, c_name,
                 retriever_weight, reader_weight,
                 question, original_text, answer, target_text="dont know"):
        self.reader = Reader(reader_name)
        self.retriever = Retriever(q_name, c_name)
        self.original_text = original_text
        self.retri_clean_reuslt = self.retriever(question, [original_text])
        self.reader_clean_result = self.reader(question, [original_text], answer)
        # print("Original retri-score: ", self.retri_clean_reuslt)
        # print("Original reader-score: ", self.reader_clean_result)
        
        self.target_text = target_text
        self.answer = answer
        
        self.retriever_weight = retriever_weight
        self.reader_weight = reader_weight
    
    
    def __call__(self, question, contexts, answer):
        
        retrieval_result = self.retriever(question, contexts)
        reader_result = self.reader(question, contexts, answer)
        # print("Contexts: ", contexts)
        retri_scores = self.retri_clean_reuslt / retrieval_result
        reader_scores = reader_result / self.reader_clean_result
        
        weighted_result = self.retriever_weight * retri_scores + self.reader_weight * reader_scores
        return weighted_result, retri_scores, reader_scores


if __name__ == "__main__":
    # fitness = Reader(model_name="Llama-7b")
    # question = "When Khoa become researcher?"
    # contexts = ["Khoa developed a strong passion for artificial intelligence during his university years. After graduating with honors, he decided to pursue a career in research. In 2025, Khoa officially became a researcher at a leading technology institute. Since then, he has contributed to several groundbreaking projects in computer vision and natural language processing.",
    #             "dog",
    #             "cat"]
    # answers = '2025'
    # reader_scores = fitness(question, contexts, answers)
    # fitness = Retriever("facebook/dpr-question_encoder-multiset-base", 
    #                     "facebook/dpr-ctx_encoder-multiset-base")
    # retriever_scores = fitness(question, contexts)
    # print(retriever_scores)
    # print(reader_scores)
    question = "When Khoa become researcher?"
    original_text = "Khoa developed a strong passion for artificial intelligence during his university years. After graduating with honors, he decided to pursue a career in research. In 2025, Khoa officially became a researcher at a leading technology institute. Since then, he has contributed to several groundbreaking projects in computer vision and natural language processing."
    contexts = ['Khoa developed a strong passion fo5 artificial intelligence durkng his university years. After graduating s ith uonors,, he decidrd to pursue a career in resexrch.. In 2052,, Khoa off9cially became a researcher ay a leading tech n;logy insgitute.. Since fhen,, he has contributed to several groundbreaking projects in compu6er v8sion and natural lan guage prcoessing..', 
                'Khoa developed a strong passion fo5 artificial intelligence durkng his university years. After gra duating sith uonors,, he decidrd to pursue a career in researxh.. In 2052,, Khoa off9cially became a researcher ay a lea ding techn;logy insgitute.. Since fhen,, he has contributed to several groundbreaking projects in compu6er v8sion and na tural language pfocessing..', 
                'Khoa developed a strong passion fo5 artificial intelligence durkng his university years. After graduating sith uonors,, he decidrd to pursue a career in resexrch.. In 2052,, Khoa off9cially became a researcher ay a leading techn;logy insgitute.. Since fhen,, he has contributed to several groundbreaking projects in ckmputer v8si on and natural language prcoessing..', 
                'Khoa developed a strong passion fo5 artificial intelligence durkng his universit y years. After graduating sith uonors,, he decidrd to pursue a career in researxh.. In 2052,, Khoa off9cially became a r esearcher ay a leading techn;logy insgitute.. Since fhen,, he has contributed to several groundbreaking projects in comp u6er v8sion and natural language prcoessing..']

    answer = "2025"
    fitness = WeightedSUm("Llama-7b", 
                          "facebook/dpr-question_encoder-multiset-base", 
                          "facebook/dpr-ctx_encoder-multiset-base",
                          0.5, 0.5,
                          question, original_text, answer) 
    
    dual_scores = fitness(question, contexts, answer)
    # print(dual_scores)
    
    # for context in contexts:
    #     dual_scores = fitness(question, [context], answer)
    #     # print(dual_scores)