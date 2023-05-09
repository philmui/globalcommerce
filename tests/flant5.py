import numpy as np
import matplotlib.pyplot as plt
from transformers import T5Tokenizer, T5ForConditionalGeneration

model_path = "/Users/pmui/models/flan-t5-xl"
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path, device_map="auto")

def inference(input_text):
  input_ids = tokenizer(input_text, return_tensors="pt").input_ids
  outputs = model.generate(input_ids, max_length=200, bos_token_id=0)
  result = tokenizer.decode(outputs[0], skip_special_tokens=True)
  print(result)

input_text = "What is the tallest building in the world?"
inference(input_text)