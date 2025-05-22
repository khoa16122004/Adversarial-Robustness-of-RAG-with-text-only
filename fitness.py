from reader import Reader
from retrieval import Retriever
import numpy as np


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
    def __init__(self, retriever_name, q_name, c_name,
                 retriever_weight, reader_weight,
                 question, original_text, answer, target_answer="dont know"):
        self.reader = Reader(retriever_name)
        self.retriever = Retriever(q_name, c_name)
        self.original_text = original_text
        self.retri_clean_reuslt = self.retriever(question, [original_text])
        self.target_text = self.target_text
        
        # print(self.retri_clean_reuslt, self.reader_clean_result)
        # raise
        
        self.retriever_weight = retriever_weight
        self.reader_weight = reader_weight
    
    
    def __call__(self, question, contexts, answer):
        
        retrieval_result = self.retriever(question, contexts) / self.retri_clean_reuslt
        reader_result = self.reader(question, contexts, self.target_text) / self.reader(question, contexts, answer)
        weighted_result = self.retriever_weight * retrieval_result + self.reader_weight * reader_result
        return weighted_result, retrieval_result, reader_result


# if __name__ == "__main__":
#     fitness = Reader(model_name="Llama-7b")
#     question = "When Khoa become researcher?"
#     contexts = ["Khoa developed a strong passion for artificial intelligence during his university years. After graduating with honors, he decided to pursue a career in research. In 2025, Khoa officially became a researcher at a leading technology institute. Since then, he has contributed to several groundbreaking projects in computer vision and natural language processing.",
#                 "dog",
#                 "cat"]
#     answers = '2025'
#     reader_scores = fitness(question, contexts, answers)
#     fitness = Retriever("facebook/dpr-question_encoder-multiset-base", 
#                         "facebook/dpr-ctx_encoder-multiset-base")
#     retriever_scores = fitness(question, contexts)
#     print(retriever_scores)
#     print(reader_scores)
    
#     fitness = FitnessDual("Llama-7b", 
#                           "facebook/dpr-question_encoder-multiset-base", 
#                           "facebook/dpr-ctx_encoder-multiset-base") 
    
#     dual_scores = fitness(question, contexts, answers)
#     print(dual_scores)