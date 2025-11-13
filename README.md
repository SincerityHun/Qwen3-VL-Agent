# Qwen3-VL Client-Server Deployment Guide

## 📋 Overview

This implementation separates Qwen3-VL inference into client and server components:

- **Client**: Preprocessing + Vision Encoder + Gradio UI (Port 7860)
- **Server**: LLM Inference with GPU acceleration (Port 8000)

## 🏗️ Architecture

```
┌─────────────────────────────────────┐
│         Client Container            │
│  ┌───────────────────────────────┐  │
│  │     Gradio Web UI (7860)      │  │
│  └───────────────────────────────┘  │
│              ↓                       │
│  ┌───────────────────────────────┐  │
│  │   Vision Preprocessing        │  │
│  │   (qwen-vl-utils)             │  │
│  └───────────────────────────────┘  │
│              ↓                       │
│  ┌───────────────────────────────┐  │
│  │   Vision Encoder (ViT)        │  │
│  │   (2-3GB model)               │  │
│  └───────────────────────────────┘  │
│              ↓                       │
│       vision_embeddings              │
└──────────────┼──────────────────────┘
               │ HTTP POST
               │ /api/v1/generate
               ↓
┌─────────────────────────────────────┐
│         Server Container            │
│  ┌───────────────────────────────┐  │
│  │   FastAPI Server (8000)       │  │
│  └───────────────────────────────┘  │
│              ↓                       │
│  ┌───────────────────────────────┐  │
│  │   LLM Inference               │  │
│  │   (Language Model Only)       │  │
│  │   GPU Accelerated             │  │
│  └───────────────────────────────┘  │
│              ↓                       │
│       generated_text                 │
└─────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose installed
- NVIDIA GPU with CUDA support (for server)
- NVIDIA Container Toolkit installed

### 1. Build and Run with Docker Compose

```bash
# Build both containers
docker-compose build

# Start services (server first, then client)
docker-compose up -d

# Check logs
docker-compose logs -f

# Access Gradio UI
# Open browser: http://localhost:7860
```

### 2. Environment Configuration

Edit `docker-compose.yml` to customize:

```yaml
services:
  server:
    environment:
      - MODEL_NAME=Qwen/Qwen3-VL-2B-Instruct  # Change model here
      - DEVICE_MAP=auto                       # GPU allocation
      - TORCH_DTYPE=auto                      # fp16/fp32
  
  client:
    environment:
      - SERVER_URL=http://server:8000         # Server endpoint
      - MODEL_NAME=Qwen/Qwen3-VL-2B-Instruct  # Must match server
```

### 3. Manual Build (without Docker Compose)

#### Server
```bash
cd server
docker build -t qwen3vl-server .
docker run --gpus all -p 8000:8000 \
  -e MODEL_NAME=Qwen/Qwen3-VL-2B-Instruct \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  qwen3vl-server
```

#### Client
```bash
cd client
docker build -t qwen3vl-client .
docker run -p 7860:7860 \
  -e SERVER_URL=http://localhost:8000 \
  -e MODEL_NAME=Qwen/Qwen3-VL-2B-Instruct \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  qwen3vl-client
```

## 📁 Project Structure

```
.
├── docker-compose.yml          # Orchestration config
│
├── client/                     # Client container
│   ├── README.md        
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── preprocessor.py         # Vision preprocessing (qwen-vl-utils)
│   ├── vision_encoder.py       # Vision Encoder extraction (2048-dim output)
│   ├── client_api.py           # HTTP client for server
│   ├── gradio_app.py           # Gradio UI
│   └── test_vision_embedding.py # Test vision encoder (images/videos)
│
├── server/                     # Server container
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── llm_inference.py        # LLM generation logic
│   └── server_api.py           # FastAPI endpoints
│
├── qwen-vl-utils/              # Vision processing utilities (But we just import this package in pip)
│   └── src/qwen_vl_utils/
│       ├── __init__.py
│       └── vision_process.py   # Image/video loading, resizing
│
├── cookbooks/                  # Example notebooks
│   ├── video_understanding.ipynb
│   ├── ocr.ipynb
│   └── ...
│
└── README/                     # Documentation
    ├── README.md               # Implementation guide
    └── CODE_MAPPING.md         # Source code mapping
```

## 🔧 API Reference

### Server Endpoints

#### 1. Health Check
```bash
GET /health

Response:
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda:0"
}
```

#### 2. Generate (Non-streaming)
```bash
POST /api/v1/generate

Request:
{
  "input_ids": [[1, 2, 3, ...]],
  "vision_embeddings": [[0.1, 0.2, ...], ...],
  "vision_token_positions": [5, 10],
  "attention_mask": [[1, 1, 1, ...]],
  "max_new_tokens": 128,
  "temperature": 0.7,
  "top_p": 0.8
}

Response:
{
  "text": "Generated response...",
  "finish_reason": "stop"
}
```

#### 3. Generate (Streaming)
```bash
POST /api/v1/generate_stream

# Returns newline-delimited JSON (NDJSON)
{"text": "Hello"}
{"text": "Hello world"}
{"text": "Hello world!", "finish_reason": "stop"}
```

## 🧪 Testing

### Test Vision Encoder (Client-Side Only)

Test vision preprocessing and encoding without requiring the server:

```bash
cd client

# Test with image
python test_vision_embedding.py image ../examples/dog.jpg

# Test with video
python test_vision_embedding.py video ../examples/IronMan.mp4

# Expected output:
# ✅ Embeddings are non-zero
# ✅ Embeddings are finite
# ✅ Vision encoder hidden dimension is 2048
# ✅ Vision token positions found
```

**Note**: Vision encoder outputs **2048-dim** embeddings, which are later projected to 3584-dim by the LLM.

### Test Server Health
```bash
curl http://localhost:8000/health
```

### Test Client Access
```bash
# Open browser
http://localhost:7860
```

### Test End-to-End
1. Upload an image in Gradio UI
2. Enter prompt: "Describe this image"
3. Click Submit
4. Check server logs: `docker-compose logs server`

## 📊 Performance Tuning

### GPU Memory Optimization

For low-memory GPUs, modify server Dockerfile:
```dockerfile
# Use 4-bit quantization
RUN pip install bitsandbytes

# In llm_inference.py
self.model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_name,
    load_in_4bit=True,
    device_map="auto"
)
```

### Client-Side Optimization

If Vision Encoder is too large for client devices:
```python
# In client_api.py, send pixel_values instead of embeddings
# Skip vision_encoder.py entirely
```

## 🛠️ Troubleshooting

### Issue: Server not starting
```bash
# Check GPU availability
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# Check logs
docker-compose logs server
```

### Issue: Client can't connect to server
```bash
# Test server from client container
docker exec qwen3vl-client curl http://server:8000/health

# Check network
docker network inspect qwen3vl_qwen3vl-network
```

### Issue: Out of memory
```bash
# Reduce model size or use quantization
# Set in docker-compose.yml:
environment:
  - MODEL_NAME=Qwen/Qwen3-VL-2B-Instruct  # Use 2B instead of 7B
```

## 📝 Development Mode

Run without Docker for development:

### Client Setup
```bash
cd client

# Create virtual environment with uv
uv venv -p 3.12
source .venv/bin/activate

# Install PyTorch with CUDA support
uv pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu128

# Install dependencies
uv pip install -r requirements.txt

# Test vision encoder
python test_vision_embedding.py image ../examples/dog.jpg
python test_vision_embedding.py video ../examples/IronMan.mp4

# Run Gradio app
export SERVER_URL=http://localhost:8000
python gradio_app.py
```

**Important Dependencies:**
- `qwen-vl-utils`: Vision processing utilities
- `decord`: Video decoding (preferred backend, more stable than torchvision)
- `transformers>=4.57.0`: Qwen3-VL model support

### Server Setup
```bash
cd server
pip install -r requirements.txt
python server_api.py
```

### Video Processing Notes

The client uses `qwen-vl-utils` for video processing with the following backends (in order of preference):
1. **decord** (recommended) - Most stable, install with `pip install decord`
2. torchcodec - Experimental
3. torchvision - Deprecated, avoid

If you encounter video processing errors, ensure `decord` is installed in your environment.

## 🔐 Production Deployment

For production, add:

1. **HTTPS/TLS**: Use nginx reverse proxy
2. **Authentication**: Add API keys to FastAPI
3. **Rate Limiting**: Use FastAPI middleware
4. **Monitoring**: Add Prometheus metrics
5. **Load Balancing**: Deploy multiple server replicas

## 📚 References

- [Qwen3-VL Model Card](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)
- [qwen-vl-utils Documentation](https://github.com/QwenLM/Qwen-VL)
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [Gradio Docs](https://gradio.app/)

## 📄 License

See LICENSE file in repository root.
