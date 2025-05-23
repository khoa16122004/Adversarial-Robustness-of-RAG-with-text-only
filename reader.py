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
            temperature =None,
            
        )
        if self.is_decoder:
            self.tokenizer.padding_side = "left"
    
    def forward(self, question, contexts, answer): # logits scores
        inputs = [self.template.format(q=question, d=text) for text in contexts]
        labels = [answer] * len(inputs)
        print("Contexts: ", contexts)
        print("Len inputs: ", len(inputs))
        
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
        
        print("First inputids: ", input_embeddings.input_ids.shape)
        print("Second inputids: ", label_embeddings.input_ids.shape)
        
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
   
    def _cal_label_logprob(self, logits, labels):
        # logits: (B, T, V), labels: (B, T)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        label_log_probs = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)
        mask = (labels != self.tokenizer.pad_token_id)
        total_log_probs = (label_log_probs * mask).sum(dim=1)
        return total_log_probs.tolist()  # return list of log probs

    def get_scores(self, input_ids, label_ids):
        scores = []
        for i in range(input_ids.size(0)):
            input_ = input_ids[i].unsqueeze(0).cuda()
            label_ = label_ids[i].unsqueeze(0).cuda()

            full_input = torch.cat([input_, label_[:, :-1]], dim=1)  # exclude last label token
            with torch.no_grad():
                logits = self.model(input_ids=full_input).logits[:, -label_.size(1):, :]
            logp = self._cal_label_logprob(logits, label_)
            scores.append(logp[0])
        return np.array(scores)
    


if __name__ == "__main__":
    reader = Reader(model_name="Llama-7b")
    question = "When Khoa become researcher?"
    contexts = ['Khoa developed a strong passion fo5 artificial intelligence durkng his university years. After gra duating sith uonors,, he decidrd to pursue a career in researxh.. In 2052,, Khoa off9cially became a researcher ay a lea ding te chn;logy insgitute.. Since fhen,, he has contributed to several groundbreaking projects in compu6er v8sion and na tural language pfocessing..']
    answers = ['2025', "dog", "cat"]
    
    pred = reader.generate(question, contexts)
    scores = []
    print("Prediction: ", pred)
    for ans in answers:
        score = reader(question, contexts, ans)
        scores.append(score)
    print(scores[1] / scores[0])
    score_normalize = np.array(scores) / np.array(scores).sum()
    print("Score: ", scores)
        
    
