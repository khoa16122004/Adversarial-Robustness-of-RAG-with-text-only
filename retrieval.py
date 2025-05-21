import torch
from transformers import BertModel, XLMRobertaModel, AlbertModel, T5EncoderModel, DPRContextEncoder, DPRQuestionEncoder
from transformers import AutoTokenizer
from utils import set_seed_everything

class Retriever(torch.nn.Module):
    def __init__(self, model_name):
        super().__init__()
        set_seed_everything(222520691)
        self.model_name = model_name
        self.tokenizer, self.d_encoder, self.q_encoder = self.load_retriever()
        self.tokenizer_kwargs = {
            "max_length": 512,
            "truncation":True,
            "padding":True, 
            "return_tensors":"pt"
        }

    def load_retriever(self):
        if "contriever" in self.model_name:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            d_encoder = Contriever.from_pretrained(self.model_namme).to("cuda")
            q_encoder = d_encoder
            
        elif "dpr" in self.model_name:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            d_encoder = DPRC.from_pretrained(self.model_name).to("cuda")
            q_encoder = DPRQ.from_pretrained(self.model_name).to("cuda")
            
        d_encoder.eval()
        q_encoder.eval()
        return tokenizer, d_encoder, q_encoder
            
    def forward(self, queries, contexts): 
        """
        queries and contexts are input ids
        """
        query_ids = self.tokenizer(queries, **self.tokenizer_kwargs)
        context_ids = self.tokenizer(contexts, **self.tokenizer_kwargs)
        query_ids.to(self.q_encoder.device)
        context_ids.to(self.q_encoder.device)

        query_embeddings = self.encode(query_ids, mode="query")
        context_embeddings = self.encode(context_ids, mode="context")
      
        scores = [q @ c for q, c in zip(query_embeddings, context_embeddings)]
        return scores
    
    def encode(self, contexts, mode="context"):
        if mode == "context":
            embedding = self.d_encoder(**contexts)
        elif mode == "query":
            embedding = self.q_encoder(**contexts)
        
        embedding = torch.nn.functional.normalize(embedding, dim=-1)    
        return self.q_encoder(**contexts)

    
class DPRC(DPRContextEncoder):
    def forward(self, **kwargs):
        output = super().forward(**kwargs)
        return output.pooler_output


class DPRQ(DPRQuestionEncoder):
    def forward(self, **kwargs):
        output = super().forward(**kwargs)
        return output.pooler_output

class Contriever(BertModel):
    def __init__(self, config, pooling="average", **kwargs):
        super().__init__(config, add_pooling_layer=False)
        if not hasattr(config, "pooling"):
            self.config.pooling = pooling
            
            
if __name__ == "__main__":
    retriever = Retriever("facebook/dpr-ctx_encoder-single-nq-base")
    queries = "What is the name of the dog"
    contexts = ["The dog's name is Max", "The cat is named Whiskers", "The bird is called Tweety"]
    scores = retriever(queries, contexts)
    print(scores)