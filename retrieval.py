import torch
from transformers import BertModel, DPRContextEncoder, DPRQuestionEncoder, AutoTokenizer
from utils import set_seed_everything

class Retriever(torch.nn.Module):
    def __init__(self, model_name):
        super().__init__()
        set_seed_everything(222520691)
        self.model_name = model_name
        self.tokenizer, self.d_encoder, self.q_encoder = self.load_retriever()
        self.tokenizer_kwargs = {
            "max_length": 512,
            "truncation": True,
            "padding": True,
            "return_tensors": "pt"
        }

    def load_retriever(self):
        if "contriever" in self.model_name:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            d_encoder = Contriever.from_pretrained(self.model_name).to("cuda")
            q_encoder = d_encoder
        elif "dpr" in self.model_name:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            d_encoder = DPRC.from_pretrained(self.model_name).to("cuda")
            q_encoder = DPRQ.from_pretrained(self.model_name).to("cuda")
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
    retriever = Retriever("facebook/dpr-ctx_encoder-single-nq-base")
    queries = ["What is the name of the dog"]
    contexts = [
        "the dog name is Max",
        "cat",
        "bird"
    ]
    scores = retriever(queries, contexts)
    print(scores)
