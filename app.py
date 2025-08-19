from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI()

tokenizer = AutoTokenizer.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    cache_dir="./Mistral-7B-v0.1",
    use_fast=False                   # Force slow tokenizer!
)
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    cache_dir="./Mistral-7B-v0.1"
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 128
    temperature: float = 1.0
    top_p: float = 0.95

@app.post("/generate")
def generate(request: GenerateRequest):
    inputs = tokenizer(request.prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            do_sample=True
        )
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"generated_text": generated}
