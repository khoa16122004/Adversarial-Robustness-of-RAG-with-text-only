from transformers import BartTokenizer, BartForConditionalGeneration
import torch

model_name = "sshleifer/distilbart-cnn-6-6"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

text = """The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct."""

input_ids = tokenizer(text, return_tensors="pt").input_ids

decoder_input_ids = [model.config.decoder_start_token_id]
predicted_ids = []
for i in range(20): 
    outputs = model(input_ids=input_ids, decoder_input_ids=torch.tensor([decoder_input_ids]))
    logits = outputs.logits[:,i,:]
    # perform argmax on the last dimension (i.e. greedy decoding)
    predicted_id = logits.argmax(-1)
    predicted_ids.append(predicted_id.item())
    print(tokenizer.decode([predicted_id.squeeze()]))
    # add predicted id to decoder_input_ids
    decoder_input_ids = decoder_input_ids + [predicted_id]