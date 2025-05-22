from reader import Reader
from retrieval import Retriever

class FitnessReader:
    def __init__(self, reader_name, original_answer: str):
        self.reader = Reader(reader_name)

    def __call__(self, question, contexts, answer):
        # Gọi reader với args phù hợp
        return self.reader(question, contexts, answer)

class FitnessRetriever:
    def __init__(self, retriever_name):
        self.retriever = Retriever(retriever_name)

    def __call__(self, *args, **kwargs):
        # Gọi retriever với args phù hợp
        return self.retriever(*args, **kwargs)

class FitnessDual:
    def __init__(self, retriever_name, reader_name):
        self.reader = Reader(reader_name)
        self.retriever = Retriever(retriever_name)

    def __call__(self, questions, contexts, answer):
        
        retrieval_result = self.retriever(questions, contexts)
        reader_result = self.reader(questions, contexts, answer)
        return retrieval_result, reader_result
    
    def weighted_sum(self, retrieval_result, reader_result, w, h):
        return w * retrieval_result + h * reader_result


if __name__ == "__main__":
    fitness = Reader(model_name="Llama-7b")
    question = "When Khoa become researcher?"
    contexts = ["Khoa developed a strong passion for artificial intelligence during his university years. After graduating with honors, he decided to pursue a career in research. In 2025, Khoa officially became a researcher at a leading technology institute. Since then, he has contributed to several groundbreaking projects in computer vision and natural language processing.",
                "dog",
                "cat"]
    answers = '2025'
    scores = fitness(question, contexts, answers)
    
    print(scores)