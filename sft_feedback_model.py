# feedback_model.py
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

BASE_MODEL = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"
LORA_PATH = "has_adapters"

# Load tokenizer from base model
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    torch_dtype=torch.float16
)

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, LORA_PATH)

# OPTIONAL but recommended: merge LoRA weights into base for inference
try:
    model = model.merge_and_unload()
except Exception:
    pass

model.eval()


def generate_feedback(password: str) -> str:
    prompt = (
        "You are a cybersecurity assistant. The user entered a weak password.\n"
        f"Password: {password}\n"
        "Explain the weaknesses and give clear, actionable advice to make it stronger.\n"
        "Keep the response short.\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            temperature=0.3,
            top_p=0.9
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)
