# Load model directly


#give it the instructions to provide feedback on creating a better password, given weak chosen password
#supervise fine tune it on dataset of bad/good/leaked password examples
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B-Instruct-bnb-4bit")
model = AutoModelForCausalLM.from_pretrained("unsloth/Llama-3.2-1B-Instruct-bnb-4bit")
messages = [
    {"role": "user", "content": "Who are you?"},
]
inputs = tokenizer.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=40)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))