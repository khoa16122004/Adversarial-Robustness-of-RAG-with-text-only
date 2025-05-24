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
    
    @torch.no_grad()
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
    
    @torch.no_grad()
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
    
    @torch.no_grad()
    def _cal_label_prob(self, probs, labels):
        result = []
        for prob, label in zip(probs, labels):
            mask = label > 0
            prob, label = prob[mask], label[mask]
            log_softmax = torch.nn.functional.log_softmax(prob, dim=-1)
            nll = -log_softmax.gather(1, label.unsqueeze(0).transpose(0, 1))
            avg_nll = torch.mean(nll)
            result.append(float(torch.exp(-avg_nll)))
        return np.array(result)

    
    @torch.no_grad()
    def get_scores(self, input_ids, label_ids):
        print("Input ids shape: ", input_ids.shape)
        print("Label ids shape: ", label_ids.shape)
        # if input_ids.shape[1] != label_ids.shape[1]:
        #     min_len = min(input_ids.shape[1], label_ids.shape[1])
        #     input_ids = input_ids[:, :min_len]
        #     label_ids = label_ids[:, :min_len]
        # raise

        outputs = self.model(
            input_ids=input_ids.to(self.model.device),
            attention_mask=(input_ids != self.tokenizer.pad_token_id).to(self.model.device),
            # labels=label_ids.to(self.model.device)
        )
        print("label ids: ", label_ids)
        print("Input ids: ", torch.argmax(outputs.logits, dim=-1))
        
        raise
  
        scores = self._cal_label_prob(outputs.logits, label_ids.to(self.model.device))
        scores = scores * 100

        return scores
    


if __name__ == "__main__":
    reader = Reader(model_name="Llama-7b")
    
    question = "When did Khoa become a researcher?"
    contexts = ['Khoa developed a strong passion for artificial intelligence during his university years. After graduating with honors, he decided to pursue a career in research. In 2025, Khoa officially became a researcher at a leading technology institute. Since then, he has contributed to several groundbreaking projects in computer vision and natural language processing.']
    
    
    inputs = [reader.template.format(q=question, d=text) for text in contexts]
    # Tokenize inputs
    input_enc = reader.tokenizer.encode(
        inputs,
        bos=True,
        eos=False,
    )
    input_ids = torch.tensor([input_enc], dtype=torch.long).to("cuda")

    max_gen_len = 50
    eos_token_id = reader.tokenizer.eos_id
    pad_token_id = reader.tokenizer.pad_id

    with torch.no_grad():
        for _ in range(max_gen_len):
            cur_len = input_ids.shape[1]
            logits = reader.model(input_ids[:, -1:], start_pos=cur_len - 1)
            next_token = torch.argmax(logits[:, -1], dim=-1).unsqueeze(0)  # (1, 1)

            input_ids = torch.cat([input_ids, next_token], dim=1)

            if next_token.item() == eos_token_id:
                break

    # Bỏ prompt ra nếu bạn chỉ muốn câu trả lời
    gen_text = reader.tokenizer.decode(input_ids[0][len(input_enc):].tolist())
    print("Generated answer:", gen_text)


            
    
