import torch
from transformers import BertModel, DPRContextEncoder, DPRQuestionEncoder, AutoTokenizer
from utils import set_seed_everything
import numpy as np


class Retriever(torch.nn.Module):
    def __init__(self, q_name, c_name):
        super().__init__()
        set_seed_everything(222520691)
        self.q_name = q_name
        self.c_name = c_name
        self.tokenizer, self.d_encoder, self.q_encoder = self.load_retriever()
        self.tokenizer_kwargs = {
            "max_length": 512,
            "truncation": True,
            "padding": True,
            "return_tensors": "pt"
        }

    def load_retriever(self):
        if "contriever" in self.q_name:
            tokenizer = AutoTokenizer.from_pretrained(self.q_name)
            q_encoder = Contriever.from_pretrained(self.c_name).to("cuda")
            d_encoder = q_encoder
        elif "dpr" in self.q_name:
            tokenizer = AutoTokenizer.from_pretrained(self.q_name)
            d_encoder = DPRC.from_pretrained(self.c_name).to("cuda")
            q_encoder = DPRQ.from_pretrained(self.q_name).to("cuda")
        else:
            raise ValueError("Unsupported model name")
        d_encoder.eval()
        q_encoder.eval()
        return tokenizer, d_encoder, q_encoder

    def forward(self, queries, contexts):
        query_ids = self.tokenizer(queries, **self.tokenizer_kwargs).to(self.q_encoder.device)
        context_ids = self.tokenizer(contexts, **self.tokenizer_kwargs).to(self.d_encoder.device)

        query_embeddings = self.encode(query_ids, mode="query")     # [B, D]
        context_embeddings = self.encode(context_ids, mode="context")  # [B, D]

        # Cosine similarity (batch-wise)
        scores = torch.matmul(query_embeddings, context_embeddings.T)
        return scores

    def encode(self, inputs, mode="context"):
        with torch.no_grad():
            if mode == "context":
                embedding = self.d_encoder(**inputs)
            elif mode == "query":
                embedding = self.q_encoder(**inputs)
            else:
                raise ValueError("Mode must be 'query' or 'context'")
        embedding = torch.nn.functional.normalize(embedding, dim=-1)
        return embedding


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

    def forward(self, **kwargs):
        output = super().forward(**kwargs)
        last_hidden = output.last_hidden_state  # [B, T, D]
        if self.config.pooling == "average":
            return last_hidden.mean(dim=1)
        else:
            raise NotImplementedError("Unsupported pooling method")


if __name__ == "__main__":
    retriever = Retriever("dpr-question_encoder-multiset-base", 
                          "facebook/dpr-ctx_encoder-multiset-base")
    question = "When Khoa become researcher?"
    contexts = ["Khoa developed a strong passion for artificial intelligence during his university years. After graduating with honors, he decided to pursue a career in research. In 2025, Khoa officially became a researcher at a leading technology institute. Since then, he has contributed to several groundbreaking projects in computer vision and natural language processing.",
                "Khoa enjoys hiking in the mountains during his free time. Recently, he adopted a cat and named it Pixel.",
                "Recently, he adopted a cat and named it Pixel."]
    scores = retriever(question, contexts)
    print(scores / scores.sum())
