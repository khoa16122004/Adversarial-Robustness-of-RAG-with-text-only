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
                 question, original_text, answer, target_text=None):
        self.reader = Reader(reader_name)
        self.retriever = Retriever(q_name, c_name)
        self.original_text = original_text
        self.retri_clean_reuslt = self.retriever(question, [original_text])
        self.reader_clean_result = self.reader(question, [original_text], answer)
        # print("Original retri-score: ", self.retri_clean_reuslt)
        print("clean reader-score: ", self.reader_clean_result)
        
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

class Targeted_WeightedSUm:
    def __init__(self, reader_name, q_name, c_name,
                 retriever_weight, reader_weight,
                 question, original_text, answer, target_text="dont know"):
        self.reader = Reader(reader_name)
        self.retriever = Retriever(q_name, c_name)
        self.original_text = original_text
        self.retri_clean_reuslt = self.retriever(question, [original_text])
        # print("Original retri-score: ", self.retri_clean_reuslt)
        # print("Original reader-score: ", self.reader_clean_result)
        
        self.target_text = target_text
        self.answer = answer

    
    
    def __call__(self, question, contexts, answer):
        
        retrieval_result = self.retriever(question, contexts)
        reader_result_with_answer = self.reader(question, contexts, answer)
        reader_result_with_target = self.reader(question, contexts, answer)
        # print("Contexts: ", contexts)
        retri_scores = self.retri_clean_reuslt / retrieval_result
        reader_scores = reader_result_with_answer / reader_result_with_target
        
        weighted_result = self.retriever_weight * retri_scores + self.reader_weight * reader_scores
        return weighted_result, retri_scores, reader_scores

class MultiScore:
    def __init__(self, reader_name, q_name, c_name,
                 question, original_text, answer, target_text="dont know"):
        self.reader = Reader(reader_name)
        self.retriever = Retriever(q_name, c_name)
        self.original_text = original_text
        self.retri_clean_reuslt = self.retriever(question, [original_text])
        self.reader_clean_result = self.reader(question, [original_text], answer)
        
        self.target_text = target_text
        self.answer = answer
        
    
    def __call__(self, question, contexts, answer):
        
        retrieval_result = self.retriever(question, contexts)
        reader_result = self.reader(question, contexts, answer)
        # print("Contexts: ", contexts)
        retri_scores = self.retri_clean_reuslt / retrieval_result
        reader_scores = reader_result / self.reader_clean_result
        
        return retri_scores, reader_scores
class Targeted_MultiScore:
    def __init__(self, reader_name, q_name, c_name,
                 question, original_text, answer, target_text="dont know"):
        self.reader = Reader(reader_name)
        self.retriever = Retriever(q_name, c_name)
        self.original_text = original_text
        self.retri_clean_reuslt = self.retriever(question, [original_text])
        
        self.target_text = target_text
        self.answer = answer
        
            
    def __call__(self, question, contexts, answer):
        
        retrieval_result = self.retriever(question, contexts)
        reader_result_with_answer = self.reader(question, contexts, answer)
        reader_result_with_target = self.reader(question, contexts, answer)
        retri_scores = self.retri_clean_reuslt / retrieval_result
        reader_scores = reader_result_with_answer / reader_result_with_target
        
        return retri_scores, reader_scores


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
    question = "What is the fastest land animal?"
    context = "The cheetah is the fastest land animal, capable of reaching speeds up to 70 mph. It has a slender build and distinctive spotted coat. Cheetahs primarily hunt gazelles and other small antelopes in Africa."
    adv_contexts = ["The cheetah is the fastest land animal, capable o r speeds up to 70 mph. It has a r d and distinctive spotted coat. h primarily hunt gazelles and h l antelopes n Africa."]

    answer = "Cheetah"
    fitness = WeightedSUm("Llama-7b", 
                          "facebook/dpr-question_encoder-multiset-base", 
                          "facebook/dpr-ctx_encoder-multiset-base",
                          0.5, 0.5,
                          question, context, answer) 
    
    fitness(question, adv_contexts, answer)
    reader = Reader(model_name="Llama-7b")
    output = reader(question, [context], answer)
    print(output)