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
        """
        Custom greedy decoding implementation
        Args:
            prompt: str - input prompt
            max_gen_len: int - maximum number of tokens to generate
        Returns:
            str - generated text
        """
        # Tokenize input prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
        
        # Initialize generation variables
        generated = input_ids.clone()
        eos_token_id = self.tokenizer.eos_token_id
        
        print(f"Starting generation with prompt length: {input_ids.shape[1]}")
        
        for step in range(max_gen_len):
            # Forward pass through model
            with torch.no_grad():
                outputs = self.model(generated)
                logits = outputs.logits
            
            # Get logits for the last token
            next_token_logits = logits[0, -1, :]  # (vocab_size,)
            
            # Greedy selection: choose token with highest probability
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0).unsqueeze(0)  # (1, 1)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token_id], dim=1)
            
            # Check for EOS token
            if next_token_id.item() == eos_token_id:
                print(f"EOS token reached at step {step}")
                break
                
            # Optional: print current token for debugging
            current_token = self.tokenizer.decode(next_token_id[0], skip_special_tokens=True)
            print(f"Step {step}: Generated token '{current_token}' (ID: {next_token_id.item()})")
        
        # Decode the generated sequence (excluding the original prompt)
        generated_tokens = generated[0, input_ids.shape[1]:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return generated_text
    
    @torch.no_grad()
    def greedy_decode_batch(self, prompts, max_gen_len=50):
        """
        Batch greedy decoding for multiple prompts
        Args:
            prompts: list of str - input prompts
            max_gen_len: int - maximum number of tokens to generate
        Returns:
            list of str - generated texts
        """
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
        
        # Track original lengths for each prompt
        original_lengths = attention_mask.sum(dim=1)
        
        # Initialize generation
        generated = input_ids.clone()
        eos_token_id = self.tokenizer.eos_token_id
        
        # Track which sequences are still generating
        not_finished = torch.ones(batch_size, dtype=torch.bool, device=self.model.device)
        
        for step in range(max_gen_len):
            if not not_finished.any():
                break
                
            # Create attention mask for current sequence
            current_attention_mask = (generated != self.tokenizer.pad_token_id).long()
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(
                    input_ids=generated,
                    attention_mask=current_attention_mask
                )
                logits = outputs.logits
            
            # Get next token for each sequence
            next_token_logits = logits[:, -1, :]  # (batch_size, vocab_size)
            next_tokens = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)  # (batch_size, 1)
            
            # Only update sequences that haven't finished
            next_tokens = next_tokens * not_finished.unsqueeze(1) + \
                         self.tokenizer.pad_token_id * (~not_finished).unsqueeze(1)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_tokens], dim=1)
            
            # Update finished status
            not_finished = not_finished & (next_tokens.squeeze(1) != eos_token_id)
        
        # Decode generated sequences
        results = []
        for i in range(batch_size):
            original_len = original_lengths[i].item()
            generated_tokens = generated[i, original_len:]
            
            # Remove padding tokens
            if self.tokenizer.pad_token_id in generated_tokens:
                pad_idx = (generated_tokens == self.tokenizer.pad_token_id).nonzero()[0].item()
                generated_tokens = generated_tokens[:pad_idx]
                
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            results.append(generated_text)
        
        return results

    @torch.no_grad()
    def forward(self, question, contexts, answer):
        """
        Tính score cho answer với từng context
        """
        inputs = [self.template.format(q=question, d=text) for text in contexts]
        print("Question:", question)
        print("Answer:", answer)
        print("Number of contexts:", len(contexts))
        
        # Tokenize inputs (prompts)
        input_embeddings = self.tokenizer(
            inputs,
            max_length=512,
            truncation=True,
            padding=True, 
            return_tensors="pt",
        )
        
        # Tokenize answer (same answer for all contexts)
        answer_embeddings = self.tokenizer(
            [answer] * len(inputs),  # repeat answer for each input
            max_length=128,  # answer thường ngắn hơn
            truncation=True,
            padding=True, 
            return_tensors="pt",
        )
        
        print("Input embeddings shape:", input_embeddings.input_ids.shape)
        print("Answer embeddings shape:", answer_embeddings.input_ids.shape)
        
        # Tính scores
        scores = self.get_scores(input_embeddings.input_ids, answer_embeddings.input_ids)
        
        print("Scores:", scores)
        return scores
    
    @torch.no_grad()
    def generate(self, question, contexts):
        """
        Original generate method using transformers generate()
        """
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
        
        # Tokenize prompt và answer
        prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
        answer_ids = self.tokenizer.encode(answer, add_special_tokens=False, return_tensors="pt").to(self.model.device)
        
        full_sequence = torch.cat([prompt_ids, answer_ids], dim=1)
        
        outputs = self.model(full_sequence)
        logits = outputs.logits[0] 
        prompt_len = prompt_ids.shape[1]
        answer_logits = logits[prompt_len-1:prompt_len-1+answer_ids.shape[1]]  # (answer_len, vocab_size)
        
        log_probs = torch.nn.functional.log_softmax(answer_logits, dim=-1)
        
        answer_tokens = answer_ids.squeeze(0)  
        token_log_probs = log_probs.gather(1, answer_tokens.unsqueeze(1)).squeeze(1)
        
        total_log_prob = token_log_probs.sum()  
        probability = torch.exp(total_log_prob).item()
        
        return probability

    @torch.no_grad()
    def get_scores(self, input_ids, answer_ids):
        batch_size = input_ids.shape[0]
        scores = []
        
        for i in range(batch_size):
            input_seq = input_ids[i:i+1] 
            answer_seq = answer_ids[i]    
            
            
            answer_seq = answer_seq[answer_seq != self.tokenizer.pad_token_id]
            
            if len(answer_seq) == 0: 
                scores.append(0.0)
                continue
            
            full_seq = torch.cat([
                input_seq.squeeze(0), 
                answer_seq
            ], dim=0).unsqueeze(0)  
            
            outputs = self.model(
                input_ids=full_seq.to(self.model.device),
                attention_mask=(full_seq != self.tokenizer.pad_token_id).to(self.model.device)
            )
            
            input_len = input_seq.shape[1]  
            answer_logits = outputs.logits[0, input_len-1:input_len-1+len(answer_seq)]  # (answer_len, vocab_size)
            
            log_probs = torch.nn.functional.log_softmax(answer_logits, dim=-1)
            
            token_log_probs = log_probs.gather(1, answer_seq.unsqueeze(1).cuda()).squeeze(1)  # (answer_len,)
            
            avg_log_prob = token_log_probs.mean()
            prob = torch.exp(avg_log_prob).item()
            
            scores.append(prob)
            
            # # Debug info
            # print(f"Sample {i}:")
            # print(f"  Input length: {input_len}")
            # print(f"  Answer length: {len(answer_seq)}")
            # print(f"  Answer tokens: {answer_seq.tolist()}")
            # print(f"  Answer text: '{self.tokenizer.decode(answer_seq, skip_special_tokens=True)}'")
            # print(f"  Token log probs: {token_log_probs.tolist()}")
            # print(f"  Average log prob: {avg_log_prob.item():.4f}")
            # print(f"  Final probability: {prob:.6f}")
            # print()
        
        return np.array(scores)


# if __name__ == "__main__":
#     reader = Reader(model_name="Llama-7b")

#     question = "When did Khoa become a researcher?"
#     context = (
#         "Khoa developed a strong passion for artificial intelligence during his university years. "
#         "After graduating with honors, he decided to pursue a career in research. "
#         "In 2025, Khoa officially became a researcher at a leading technology institute. "
#         "Since then, he has contributed to several groundbreaking projects in computer vision and NLP."
#     )
#     answer = "2025"

#     print("="*60)
#     print("TESTING ANSWER PROBABILITY CALCULATION")
#     print("="*60)
    
#     prob = reader.calculate_answer_probability(question, context, answer)
#     print(f"Question: {question}")
#     print(f"Context: {context[:100]}...")
#     print(f"Answer: '{answer}'")
#     print(f"Probability: {prob:.8f}")
#     print()

#     contexts = [
#         context,
#         "Khoa started his research career in 2024 after completing his PhD.",
#         "In 2023, Khoa joined the research team and began working on AI projects."
#     ]
    
#     scores = reader.forward(question, contexts, answer)
#     for i, (ctx, score) in enumerate(zip(contexts, scores)):
#         print(f"Context {i+1}: {ctx[:50]}...")
#         print(f"Score: {score:.8f}")
#         print()

#     print("Comparing different answers:")
#     test_answers = ["2025", "2024", "2023", "never"]
#     for test_answer in test_answers:
#         prob = reader.calculate_answer_probability(question, context, test_answer)
#         print(f"Answer '{test_answer}': {prob:.8f}")
    
#     print("="*60)
#     print("TESTING GREEDY GENERATION")  
#     print("="*60)
    
#     prompt = reader.template.format(q=question, d=context)
#     generated_answer = reader.greedy_decode(prompt, max_gen_len=20)
#     print("Generated answer:", generated_answer)