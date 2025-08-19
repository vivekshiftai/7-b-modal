from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI(title="Mistral 7B API", description="A FastAPI endpoint for Mistral 7B text generation")

@app.get("/")
def root():
    return {
        "message": "Mistral 7B API is running!",
        "endpoints": {
            "generate": "POST /generate - Generate text using Mistral 7B"
        },
        "usage": {
            "method": "POST",
            "url": "/generate",
            "body": {
                "prompt": "Your input text",
                "max_new_tokens": 128,
                "temperature": 1.0,
                "top_p": 0.95
            }
        }
    }

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
    try:
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
    except Exception as e:
        return {"error": str(e)}, 500

@app.get("/generate")
def generate_get():
    return {
        "error": "This endpoint requires a POST request",
        "usage": {
            "method": "POST",
            "url": "/generate",
            "body": {
                "prompt": "Your input text",
                "max_new_tokens": 128,
                "temperature": 1.0,
                "top_p": 0.95
            }
        }
    }
