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
    
    @torch.no_grad()
    def greedy_decode(self, prompt, max_gen_len=50):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
        
        generated = input_ids.clone()
        eos_token_id = self.tokenizer.eos_token_id
        
        
        for step in range(max_gen_len):
            with torch.no_grad():
                outputs = self.model(generated)
                logits = outputs.logits
            
            next_token_logits = logits[0, -1, :]  # (vocab_size,)
            
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0).unsqueeze(0)  # (1, 1)
            
            generated = torch.cat([generated, next_token_id], dim=1)
            
            if next_token_id.item() == eos_token_id:
                print(f"EOS token reached at step {step}")
                break
                
            current_token = self.tokenizer.decode(next_token_id[0], skip_special_tokens=True)
            print(f"Step {step}: Generated token '{current_token}' (ID: {next_token_id.item()})")
        
        generated_tokens = generated[0, input_ids.shape[1]:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return generated_text
    
    @torch.no_grad()
    def greedy_decode_batch(self, prompts, max_gen_len=50):
        # Tokenize all prompts
        encoded = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.model.device)
        
        input_ids = encoded.input_ids
        attention_mask = encoded.attention_mask
        batch_size = input_ids.shape[0]
        
        original_lengths = attention_mask.sum(dim=1)
        
        generated = input_ids.clone()
        eos_token_id = self.tokenizer.eos_token_id
        
        not_finished = torch.ones(batch_size, dtype=torch.bool, device=self.model.device)
        
        for step in range(max_gen_len):
            if not not_finished.any():
                break
                
            current_attention_mask = (generated != self.tokenizer.pad_token_id).long()
            
            with torch.no_grad():
                outputs = self.model(
                    input_ids=generated,
                    attention_mask=current_attention_mask
                )
                logits = outputs.logits
            
            next_token_logits = logits[:, -1, :]  # (batch_size, vocab_size)
            next_tokens = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)  # (batch_size, 1)
            
            next_tokens = next_tokens * not_finished.unsqueeze(1) + \
                         self.tokenizer.pad_token_id * (~not_finished).unsqueeze(1)
            
            generated = torch.cat([generated, next_tokens], dim=1)
            
            not_finished = not_finished & (next_tokens.squeeze(1) != eos_token_id)
        
        results = []
        for i in range(batch_size):
            original_len = original_lengths[i].item()
            generated_tokens = generated[i, original_len:]
            
            if self.tokenizer.pad_token_id in generated_tokens:
                pad_idx = (generated_tokens == self.tokenizer.pad_token_id).nonzero()[0].item()
                generated_tokens = generated_tokens[:pad_idx]
                
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            results.append(generated_text)
        
        return results

    @torch.no_grad()
    def forward(self, question, contexts, answer):
        inputs = [self.template.format(q=question, d=text) for text in contexts]
        scores = []
        for prompt in inputs:
            scores.append(self.calculate_answer_probability(question, prompt, answer))
                
        return np.array(scores)
    
    @torch.no_grad()
    def generate(self, question, contexts):
        inputs = [self.template.format(q=question, d=text) for text in contexts]
        input_ids = self.tokenizer(
                inputs,
                max_length=512,
                truncation=True,
                padding=True, 
                return_tensors="pt",
        )
        outputs = self.model.generate(
            input_ids=input_ids.input_ids.to(self.model.device), 
            attention_mask=input_ids.attention_mask.to(self.model.device), 
            **self.generate_kwargs
        )
        outputs = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)

        if isinstance(outputs, list):
            return [o.split("Answer:")[-1].strip() for o in outputs]
        else:
            return outputs.split("Answer:")[-1].strip()
    
    @torch.no_grad()
    def calculate_answer_probability(self, question, context, answer):

        
        
        prompt = self.template.format(q=question, d=context)
        # print("Input: ", prompt)
        # print("Answer: ", answer)
        # print("question: ", question)
        prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
        answer_ids = self.tokenizer.encode(answer, add_special_tokens=False, return_tensors="pt").to(self.model.device)
        
        full_sequence = torch.cat([prompt_ids, answer_ids], dim=1)
        
        outputs = self.model(full_sequence)
        logits = outputs.logits[0] 
        prompt_len = prompt_ids.shape[1]
        answer_logits = logits[prompt_len-1:prompt_len - 1 +answer_ids.shape[1]]  # (answer_len, vocab_size)
        
        log_probs = torch.nn.functional.log_softmax(answer_logits, dim=-1)
        
        answer_tokens = answer_ids.squeeze(0)  
        token_log_probs = log_probs.gather(1, answer_tokens.unsqueeze(1)).squeeze(1)
        
        total_log_prob = token_log_probs.sum()  
        probability = torch.exp(total_log_prob).item()
        
        return probability
    
    @torch.no_grad()
    def forward_batch(self, question, contexts, answer):
        # Format all prompts
        prompts_text = [self.template.format(q=question, d=context) for context in contexts]
        batch_size = len(prompts_text)

        # Tokenize prompts (batch) - ensuring add_special_tokens=True for prompts
        # This should align with how self.tokenizer.encode(prompt_text, add_special_tokens=True) works.
        # Note: The `self.tokenizer()` call might have different default `add_special_tokens` behavior
        # than `self.tokenizer.encode()`. Explicitly set it if needed, or ensure they align.
        # Most HuggingFace tokenizers add special tokens by default with __call__.
        tokenized_prompts = self.tokenizer(
            prompts_text,
            padding=True,
            truncation=False, # To match calculate_answer_probability which doesn't truncate prompt
            return_tensors="pt",
            add_special_tokens=True # Explicitly match `encode`
        ).to(self.model.device)
        batch_prompt_ids = tokenized_prompts.input_ids
        batch_prompt_attention_mask = tokenized_prompts.attention_mask
        
        # Actual (unpadded) lengths of prompts
        prompt_lengths = batch_prompt_attention_mask.sum(dim=1) # (batch_size)

        # Tokenize answer (single) - no special tokens
        answer_ids_single = self.tokenizer.encode(
            answer,
            add_special_tokens=False, # Crucial
            return_tensors="pt"
        ).to(self.model.device) # Shape: (1, answer_len)

        if answer_ids_single.shape[1] == 0:
            # print(f"Warning: Empty answer_ids for answer: '{answer}'. All probabilities will be 0.")
            return np.zeros(batch_size, dtype=float)

        answer_len = answer_ids_single.shape[1]
        
        # Expand answer_ids to match batch size
        batch_answer_ids = answer_ids_single.expand(batch_size, -1) # (batch_size, answer_len)

        # Construct full sequences and attention masks for the batch
        batch_full_input_ids = torch.cat([batch_prompt_ids, batch_answer_ids], dim=1)
        
        # Attention mask for the answer part (all 1s)
        answer_attention_mask = torch.ones_like(batch_answer_ids, device=self.model.device)
        batch_full_attention_mask = torch.cat([batch_prompt_attention_mask, answer_attention_mask], dim=1)

        # Model forward pass
        outputs = self.model(
            input_ids=batch_full_input_ids,
            attention_mask=batch_full_attention_mask
        )
        all_logits = outputs.logits # (batch_size, full_seq_len, vocab_size)

        # Extract logits for answer tokens for each item in the batch
        # This is the trickiest part for batching due to varying prompt_lengths.
        answer_logits_list = []
        for i in range(batch_size):
            # Start index in all_logits for this item's answer predictions
            # Logits for answer_token_j are at sequence_position (prompt_lengths[i] - 1 + j)
            # Slice from (prompt_lengths[i] - 1) up to (prompt_lengths[i] - 1 + answer_len)
            start_idx = prompt_lengths[i] - 1
            end_idx = start_idx + answer_len
            current_answer_logits = all_logits[i, start_idx:end_idx, :] # (answer_len, vocab_size)
            answer_logits_list.append(current_answer_logits)
        
        answer_logits_batch = torch.stack(answer_logits_list, dim=0) # (batch_size, answer_len, vocab_size)

        # Calculate probabilities
        log_probs_batch = torch.nn.functional.log_softmax(answer_logits_batch, dim=-1)
        
        # Gather log_probs for the actual answer tokens
        # batch_answer_ids is (batch_size, answer_len)
        # Need to unsqueeze for gather: (batch_size, answer_len, 1)
        answer_target_ids_batch = batch_answer_ids.unsqueeze(2) # (batch_size, answer_len, 1)
        
        token_log_probs_batch = log_probs_batch.gather(2, answer_target_ids_batch).squeeze(2) # (batch_size, answer_len)
        
        total_log_prob_batch = token_log_probs_batch.sum(dim=1) # (batch_size)
        probabilities_batch = torch.exp(total_log_prob_batch)
        
        return probabilities_batch.cpu().numpy()
        
     




if __name__ == "__main__":
    reader = Reader(model_name="Llama-7b")

    question = "What is the fastest land animal?"
    context = "The cheetah is the fastest land animal, capable of reaching speeds up to 70 mph. It has a slender build and distinctive spotted coat. Cheetahs primarily hunt gazelles and other small antelopes in Africa."
    adv_contexts = ["The cheetah is the fastest land animal, capable of reaching speeds up to 70 mph. It has a slender build and distinctive spotted coat. Cheetahs primarily hunt gazelles and other small antelopes in Africa.",
                    "The cheetah is the fastest land animal, capable o r speeds up to 70 mph. It has a r d and distinctive spotted coat. h primarily hunt gazelles and h l antelopes n Africa."
                    ]
    answer = "Cheetah"
        
    print(reader(question, adv_contexts, answer))
    print(reader.forward_batch(question, adv_contexts, answer))