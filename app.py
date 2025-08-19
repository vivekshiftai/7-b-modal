from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
import psutil
import os
import gc
import logging
from typing import Dict, Any
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('mistral_api.log')
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Mistral 7B API", description="A FastAPI endpoint for Mistral 7B text generation")

# Global variables for metrics
model_metrics = {
    "total_requests": 0,
    "total_tokens_generated": 0,
    "total_inference_time": 0.0,
    "average_inference_time": 0.0,
    "model_loaded": False,
    "model_load_time": 0.0,
    "initial_memory_usage": 0.0
}

def get_system_metrics() -> Dict[str, Any]:
    """Get current system metrics"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    return {
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "memory_usage_mb": memory_info.rss / 1024 / 1024,
        "memory_percent": process.memory_percent(),
        "gpu_memory_allocated_mb": torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0,
        "gpu_memory_reserved_mb": torch.cuda.memory_reserved() / 1024 / 1024 if torch.cuda.is_available() else 0,
        "available_memory_mb": psutil.virtual_memory().available / 1024 / 1024,
        "total_memory_mb": psutil.virtual_memory().total / 1024 / 1024
    }

@app.get("/")
def root():
    return {
        "message": "Mistral 7B API is running!",
        "endpoints": {
            "generate": "POST /generate - Generate text using Mistral 7B",
            "metrics": "GET /metrics - Get system and model metrics",
            "health": "GET /health - Health check with basic metrics"
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

# Load model with timing
logger.info("Loading Mistral 7B model...")
model_load_start = time.time()

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

model_load_time = time.time() - model_load_start
model_metrics["model_load_time"] = model_load_time
model_metrics["model_loaded"] = True
model_metrics["initial_memory_usage"] = get_system_metrics()["memory_usage_mb"]

logger.info(f"Model loaded in {model_load_time:.2f} seconds")
logger.info(f"Device: {device}")
logger.info(f"Initial memory usage: {model_metrics['initial_memory_usage']:.2f} MB")

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 128
    temperature: float = 1.0
    top_p: float = 0.95

@app.post("/generate")
def generate(request: GenerateRequest):
    # Start timing and get initial metrics
    start_time = time.time()
    initial_metrics = get_system_metrics()
    
    logger.info(f"Processing request: prompt_length={len(request.prompt)}, max_tokens={request.max_new_tokens}")
    
    try:
        # Tokenize input
        tokenize_start = time.time()
        inputs = tokenizer(request.prompt, return_tensors="pt").to(device)
        tokenize_time = time.time() - tokenize_start
        
        # Generate text
        inference_start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=True
            )
        inference_time = time.time() - inference_start
        
        # Decode output
        decode_start = time.time()
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        decode_time = time.time() - decode_start
        
        # Calculate total time and tokens
        total_time = time.time() - start_time
        input_tokens = len(tokenizer.encode(request.prompt))
        output_tokens = len(outputs[0]) - input_tokens
        tokens_per_second = output_tokens / inference_time if inference_time > 0 else 0
        
        # Get final metrics
        final_metrics = get_system_metrics()
        
        # Update global metrics
        model_metrics["total_requests"] += 1
        model_metrics["total_tokens_generated"] += output_tokens
        model_metrics["total_inference_time"] += inference_time
        model_metrics["average_inference_time"] = model_metrics["total_inference_time"] / model_metrics["total_requests"]
        
        logger.info(f"Request completed: total_time={total_time:.3f}s, inference_time={inference_time:.3f}s, tokens_generated={output_tokens}, tokens_per_sec={tokens_per_second:.2f}")
        
        # Prepare response with metrics
        response = {
            "generated_text": generated,
            "metrics": {
                "timing": {
                    "total_time_seconds": round(total_time, 3),
                    "tokenize_time_seconds": round(tokenize_start - start_time, 3),
                    "inference_time_seconds": round(inference_time, 3),
                    "decode_time_seconds": round(decode_time, 3),
                    "tokens_per_second": round(tokens_per_second, 2)
                },
                "tokens": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens
                },
                "memory": {
                    "memory_usage_mb": round(final_metrics["memory_usage_mb"], 2),
                    "memory_increase_mb": round(final_metrics["memory_usage_mb"] - initial_metrics["memory_usage_mb"], 2),
                    "gpu_memory_allocated_mb": round(final_metrics["gpu_memory_allocated_mb"], 2),
                    "gpu_memory_reserved_mb": round(final_metrics["gpu_memory_reserved_mb"], 2)
                },
                "system": {
                    "cpu_percent": round(final_metrics["cpu_percent"], 2),
                    "available_memory_mb": round(final_metrics["available_memory_mb"], 2),
                    "memory_percent": round(final_metrics["memory_percent"], 2)
                }
            }
        }
        
        return response
        
    except Exception as e:
        error_time = time.time() - start_time
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        return {
            "error": str(e),
            "metrics": {
                "error_time_seconds": round(error_time, 3),
                "memory_usage_mb": round(get_system_metrics()["memory_usage_mb"], 2)
            }
        }

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

@app.get("/metrics")
def get_metrics():
    """Get comprehensive system and model metrics"""
    logger.debug("Metrics endpoint accessed")
    current_metrics = get_system_metrics()
    
    return {
        "model": {
            "loaded": model_metrics["model_loaded"],
            "load_time_seconds": round(model_metrics["model_load_time"], 2),
            "device": str(device),
            "cuda_available": torch.cuda.is_available()
        },
        "requests": {
            "total_requests": model_metrics["total_requests"],
            "total_tokens_generated": model_metrics["total_tokens_generated"],
            "average_inference_time_seconds": round(model_metrics["average_inference_time"], 3) if model_metrics["total_requests"] > 0 else 0
        },
        "memory": {
            "current_memory_usage_mb": round(current_metrics["memory_usage_mb"], 2),
            "initial_memory_usage_mb": round(model_metrics["initial_memory_usage"], 2),
            "memory_increase_mb": round(current_metrics["memory_usage_mb"] - model_metrics["initial_memory_usage"], 2),
            "gpu_memory_allocated_mb": round(current_metrics["gpu_memory_allocated_mb"], 2),
            "gpu_memory_reserved_mb": round(current_metrics["gpu_memory_reserved_mb"], 2),
            "memory_percent": round(current_metrics["memory_percent"], 2)
        },
        "system": {
            "cpu_percent": round(current_metrics["cpu_percent"], 2),
            "available_memory_mb": round(current_metrics["available_memory_mb"], 2),
            "total_memory_mb": round(current_metrics["total_memory_mb"], 2),
            "available_memory_percent": round((current_metrics["available_memory_mb"] / current_metrics["total_memory_mb"]) * 100, 2)
        }
    }

@app.get("/health")
def health_check():
    """Health check endpoint with basic metrics"""
    logger.debug("Health check endpoint accessed")
    current_metrics = get_system_metrics()
    
    return {
        "status": "healthy" if model_metrics["model_loaded"] else "loading",
        "model_loaded": model_metrics["model_loaded"],
        "total_requests": model_metrics["total_requests"],
        "memory_usage_mb": round(current_metrics["memory_usage_mb"], 2),
        "cpu_percent": round(current_metrics["cpu_percent"], 2),
        "gpu_memory_mb": round(current_metrics["gpu_memory_allocated_mb"], 2)
    }
