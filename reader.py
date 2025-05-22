import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from transformers import LlamaForCausalLM, LlamaTokenizer, MistralForCausalLM
import os
import json
import numpy as np
from utils import set_seed_everything
set_seed_everything(222520691)

cls_mapping = {
    "Llama-7b": (LlamaForCausalLM, LlamaTokenizer, True, "Llama-2-7b-chat-hf", "meta-llama"),
    "Llama-13b": (LlamaForCausalLM, LlamaTokenizer, True, "Llama-2-13b-chat-hf", "meta-llama"),
    "Mistral-7b": (MistralForCausalLM, AutoTokenizer, True, "Mistral-7B-Instruct-v0.2"),
    "vicuna-7b": (LlamaForCausalLM, LlamaTokenizer, True, "vicuna-7b-v1.5"),
    "vicuna-13b": (LlamaForCausalLM, LlamaTokenizer, True, "vicuna-13b-v1.5"),
    "gemma-7b": (AutoModelForCausalLM, AutoTokenizer, True, "gemma-7b-it")
}


templates = {
    "Llama-7b": "reader_template/Llama.json",
    "Llama-13b": "reader_template/Llama.json",
    "Mistral-7b": "reader_template/Mistral.json",
    "vicuna-7b": "reader_template/vicuna.json",
    "vicuna-13b": "reader_template/vicuna.json",
    "gemma-7b": "reader_template/gemma.json"
}


class Reader(torch.nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        with open(templates[model_name], "r") as f:
            self.template = json.load(f)[0]
        model_cls, tokenizer_cls, self.is_decoder, hf_name, prefix = cls_mapping[model_name]
        self.model = model_cls.from_pretrained(os.path.join(prefix, hf_name)).cuda()
        self.tokenizer = tokenizer_cls.from_pretrained(os.path.join(prefix, hf_name))
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.generate_kwargs = dict(
            max_new_tokens=30,
            return_dict_in_generate=True,
            output_hidden_states=True,
            output_scores=True,
            do_sample=False,  # greedy decoding
            top_p=None,
            temperature=None,
            
        )
        if self.is_decoder:
            self.tokenizer.padding_side = "left"
    
    def forward(self, question, contexts, answer): # logits scores
        inputs = [self.template.format(q=question, d=text) for text in contexts]
        labels = [answer] * len(inputs)
        
        input_embeddings = self.tokenizer(
            inputs,
            max_length=512,
            truncation=True,
            padding=True, 
            return_tensors="pt",
        )
        label_embeddings = self.tokenizer(
            labels, 
            max_length=512,
            truncation=True,
            padding=True, 
            return_tensors="pt",
        )
        
        scores = self.get_scores(input_embeddings.input_ids, label_embeddings.input_ids)
        return scores
    
    def generate(self, question, contexts): # text generation
        
        """
        question: str
        contexts: list of str
        """
        
        inputs = [self.template.format(q=question, d=text) for text in contexts]
        input_ids = self.tokenizer(
                inputs,
                max_length=512,
                truncation=True,
                padding=True, 
                return_tensors="pt",
        )
        outputs = self.model.generate(input_ids=input_ids.input_ids.to(self.model.device), attention_mask=input_ids.attention_mask.to(self.model.device), **self.generate_kwargs)
        outputs = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)

        if isinstance(outputs, list):
            return [o.split("Answer:")[-1].strip() for o in outputs]
        else:
            return outputs.split("Answer:")[-1].strip()
    
    def _cal_label_prob(self, probs, labels):
        result = []
        for prob, label in zip(probs, labels):
            mask = label > 0
            prob, label = prob[mask], label[mask]
            log_softmax = torch.nn.functional.log_softmax(prob, dim=-1)
            nll = -log_softmax.gather(1, label.unsqueeze(0).transpose(0, 1))
            avg_nll = torch.mean(nll)  # Sửa: lấy mean thay vì sum
            result.append(float(torch.exp(-avg_nll)))
        return np.array(result)

        
    # def _cal_label_prob(self, probs, labels):
    #     # probs: (B, N, C)  -- B: batch size, N: seq len, C: num classes
    #     # labels: (B, N)
    #     probs = probs.cuda()
    #     labels = labels.cuda()
        
    #     mask = labels > 0                                # (B, N)
    #     masked_probs = probs[mask]                       # (total_valid_positions, C)
    #     masked_labels = labels[mask]                     # (total_valid_positions)

    #     log_softmax = torch.nn.functional.log_softmax(masked_probs, dim=-1)
    #     nll = -log_softmax[torch.arange(masked_labels.shape[0], device=masked_labels.device), masked_labels]  # (total_valid_positions,)

    #     # Now group back per sample to get average NLL per sample
    #     # Step 1: build a mapping from flat index -> batch index
    #     batch_idx = torch.arange(labels.shape[0]).unsqueeze(1).expand_as(labels)  # (B, N)
    #     masked_batch_idx = batch_idx[mask]  # (total_valid_positions,)

    #     total_nll = torch.zeros(labels.shape[0], device=probs.device)
    #     count = torch.zeros(labels.shape[0], device=probs.device)

    #     total_nll.scatter_add_(0, masked_batch_idx, nll)
    #     count.scatter_add_(0, masked_batch_idx, torch.ones_like(nll))

    #     avg_nll = total_nll / count
    #     return torch.exp(avg_nll).tolist()
    
    def get_scores(self, input_ids, label_ids):
        if input_ids.shape[1] != label_ids.shape[1]:
            min_len = min(input_ids.shape[1], label_ids.shape[1])
            input_ids = input_ids[:, :min_len]
            label_ids = label_ids[:, :min_len]

        outputs = self.model(input_ids=input_ids.to(self.model.device), labels=label_ids.to(self.model.device))
        scores = self._cal_label_prob(outputs.logits, label_ids.to(self.model.device))
        scores = scores * 100

        return scores
    


if __name__ == "__main__":
    reader = Reader(model_name="Llama-7b")
    question = "When Khoa become researcher?"
    contexts = ["Khoa developed a strong passion foy artificial intelligence during his university rs.. Aftwr graduating dith honors, he ddcided to pursue aa career in r4search.. In 2025, Khoa officilaly became a researcher at a leadinf techno,ogy institute. Since then, he's has contributed to several groundbreaking projects is puter vision ane natural lengvich processing."]
    answers = ['2025', "dont know"]
    
    pred = reader.generate(question, contexts)
    scores = []
    print("Prediction: ", pred)
    for ans in answers:
        score = reader(question, contexts, ans)
        scores.append(score)
    print(scores[1] / scores[0])
    score_normalize = np.array(scores) / np.array(scores).sum()
    print("Score: ", score_normalize)
        
    
