# Mistral 7B FastAPI Endpoint

A FastAPI application that serves the Mistral 7B model for text generation.

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Mistral 7B Model
You need to download the Mistral 7B model to the `./Mistral-7B-v0.1` directory. You can do this using Hugging Face's transformers library:

```bash
python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; AutoTokenizer.from_pretrained('mistralai/Mistral-7B-v0.1', cache_dir='./Mistral-7B-v0.1'); AutoModelForCausalLM.from_pretrained('mistralai/Mistral-7B-v0.1', cache_dir='./Mistral-7B-v0.1')"
```

### 3. Start the API Server
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

### 4. Test the Endpoint
```bash
curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{"prompt":"Hello! What is AI?","max_new_tokens":64}'
```

## API Endpoints

### POST /generate
Generate text using the Mistral 7B model.

**Request Body:**
```json
{
  "prompt": "Your input text here",
  "max_new_tokens": 128,
  "temperature": 1.0,
  "top_p": 0.95
}
```

**Response:**
```json
{
  "generated_text": "Generated text response"
}
```

## Parameters

- `prompt` (required): The input text to generate from
- `max_new_tokens` (optional, default: 128): Maximum number of tokens to generate
- `temperature` (optional, default: 1.0): Controls randomness (0.0 = deterministic, 1.0 = very random)
- `top_p` (optional, default: 0.95): Nucleus sampling parameter

## System Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended for faster inference)
- At least 16GB RAM (32GB recommended)
- ~14GB disk space for the model
