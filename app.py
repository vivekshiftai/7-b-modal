from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
import psutil
import gc
from datetime import datetime
from typing import Dict, List
import json

app = FastAPI(title="Mistral 7B API", description="A FastAPI endpoint for Mistral 7B text generation")

# Global metrics storage
metrics = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "total_tokens_generated": 0,
    "total_inference_time": 0.0,
    "average_inference_time": 0.0,
    "requests_per_minute": 0,
    "memory_usage": [],
    "gpu_usage": [],
    "start_time": datetime.now().isoformat(),
    "last_request_time": None
}

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process()
    return {
        "ram_mb": process.memory_info().rss / 1024 / 1024,
        "ram_percent": process.memory_percent(),
        "timestamp": datetime.now().isoformat()
    }

def get_gpu_usage():
    """Get GPU memory usage if available"""
    if torch.cuda.is_available():
        return {
            "gpu_memory_allocated_mb": torch.cuda.memory_allocated() / 1024 / 1024,
            "gpu_memory_reserved_mb": torch.cuda.memory_reserved() / 1024 / 1024,
            "gpu_memory_percent": torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100 if torch.cuda.max_memory_allocated() > 0 else 0,
            "timestamp": datetime.now().isoformat()
        }
    return None

@app.get("/")
def root():
    return {
        "message": "Mistral 7B API is running!",
        "endpoints": {
            "generate": "POST /generate - Generate text using Mistral 7B",
            "metrics": "GET /metrics - View model performance metrics",
            "health": "GET /health - Check system health"
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
    start_time = time.time()
    metrics["total_requests"] += 1
    metrics["last_request_time"] = datetime.now().isoformat()
    
    try:
        # Record memory before generation
        memory_before = get_memory_usage()
        gpu_before = get_gpu_usage()
        
        # Tokenize input
        inputs = tokenizer(request.prompt, return_tensors="pt").to(device)
        input_tokens = inputs['input_ids'].shape[1]
        
        # Generate text
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=True
            )
        
        # Decode output
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        output_tokens = outputs.shape[1] - input_tokens
        
        # Calculate metrics
        inference_time = time.time() - start_time
        metrics["successful_requests"] += 1
        metrics["total_tokens_generated"] += output_tokens
        metrics["total_inference_time"] += inference_time
        metrics["average_inference_time"] = metrics["total_inference_time"] / metrics["successful_requests"]
        
        # Record memory after generation
        memory_after = get_memory_usage()
        gpu_after = get_gpu_usage()
        
        # Store memory metrics (keep last 100 entries)
        if len(metrics["memory_usage"]) >= 100:
            metrics["memory_usage"] = metrics["memory_usage"][-50:]
        metrics["memory_usage"].append({
            "before": memory_before,
            "after": memory_after,
            "request_id": metrics["total_requests"]
        })
        
        if gpu_before and gpu_after:
            if len(metrics["gpu_usage"]) >= 100:
                metrics["gpu_usage"] = metrics["gpu_usage"][-50:]
            metrics["gpu_usage"].append({
                "before": gpu_before,
                "after": gpu_after,
                "request_id": metrics["total_requests"]
            })
        
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return {
            "generated_text": generated,
            "metrics": {
                "inference_time_seconds": round(inference_time, 3),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "tokens_per_second": round(output_tokens / inference_time, 2) if inference_time > 0 else 0,
                "memory_usage_mb": round(memory_after["ram_mb"], 2),
                "gpu_memory_mb": round(gpu_after["gpu_memory_allocated_mb"], 2) if gpu_after else None
            }
        }
    except Exception as e:
        metrics["failed_requests"] += 1
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

@app.get("/metrics")
def get_metrics():
    """Get comprehensive model performance metrics"""
    current_memory = get_memory_usage()
    current_gpu = get_gpu_usage()
    
    # Calculate uptime
    start_time = datetime.fromisoformat(metrics["start_time"])
    uptime = datetime.now() - start_time
    
    # Calculate success rate
    success_rate = (metrics["successful_requests"] / metrics["total_requests"] * 100) if metrics["total_requests"] > 0 else 0
    
    return {
        "model_info": {
            "model_name": "Mistral-7B-v0.1",
            "device": str(device),
            "cuda_available": torch.cuda.is_available()
        },
        "performance_metrics": {
            "total_requests": metrics["total_requests"],
            "successful_requests": metrics["successful_requests"],
            "failed_requests": metrics["failed_requests"],
            "success_rate_percent": round(success_rate, 2),
            "total_tokens_generated": metrics["total_tokens_generated"],
            "average_inference_time_seconds": round(metrics["average_inference_time"], 3),
            "total_inference_time_seconds": round(metrics["total_inference_time"], 3),
            "uptime_seconds": int(uptime.total_seconds()),
            "uptime_formatted": str(uptime).split('.')[0]
        },
        "current_system_status": {
            "memory_usage_mb": round(current_memory["ram_mb"], 2),
            "memory_usage_percent": round(current_memory["ram_percent"], 2),
            "gpu_memory_allocated_mb": round(current_gpu["gpu_memory_allocated_mb"], 2) if current_gpu else None,
            "gpu_memory_percent": round(current_gpu["gpu_memory_percent"], 2) if current_gpu else None
        },
        "recent_activity": {
            "last_request_time": metrics["last_request_time"],
            "memory_history_count": len(metrics["memory_usage"]),
            "gpu_history_count": len(metrics["gpu_usage"])
        }
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    current_memory = get_memory_usage()
    current_gpu = get_gpu_usage()
    
    # Check if model is loaded
    model_loaded = model is not None and tokenizer is not None
    
    # Check memory usage (warning if > 80%)
    memory_warning = current_memory["ram_percent"] > 80
    
    # Check GPU memory (warning if > 90%)
    gpu_warning = False
    if current_gpu:
        gpu_warning = current_gpu["gpu_memory_percent"] > 90
    
    return {
        "status": "healthy" if model_loaded and not memory_warning and not gpu_warning else "warning",
        "model_loaded": model_loaded,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "memory_usage_mb": round(current_memory["ram_mb"], 2),
        "memory_usage_percent": round(current_memory["ram_percent"], 2),
        "memory_warning": memory_warning,
        "gpu_memory_mb": round(current_gpu["gpu_memory_allocated_mb"], 2) if current_gpu else None,
        "gpu_memory_percent": round(current_gpu["gpu_memory_percent"], 2) if current_gpu else None,
        "gpu_warning": gpu_warning,
        "timestamp": datetime.now().isoformat()
    }
