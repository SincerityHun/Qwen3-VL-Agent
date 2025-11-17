# Qwen3-VL Client-Server Deployment Guide

## 📋 Overview

This implementation separates Qwen3-VL inference into client and server components:

- **Client**: Preprocessing + Vision Encoder + Gradio UI (Port 7860)
- **Server**: LLM Inference with GPU acceleration (Port 8001)

## 🏗️ Baseline Architecture

```
┌─────────────────────────────────────┐
│         Client Container            │
│  ┌───────────────────────────────┐  │
│  │     Gradio Web UI (7860)      │  │
│  └───────────────────────────────┘  │
│              ↓                      │
│  ┌───────────────────────────────┐  │
│  │   Vision Preprocessing        │  │
│  │   (qwen-vl-utils)             │  │
│  └───────────────────────────────┘  │
│              ↓                      │
│  ┌───────────────────────────────┐  │
│  │   Vision Encoder (ViT)        │  │
│  │   (2-3GB model)               │  │
│  └───────────────────────────────┘  │
│              ↓                      │
│       vision_embeddings             │
└──────────────┼──────────────────────┘
               │ HTTP POST
               │ /api/v1/generate
               ↓
┌─────────────────────────────────────┐
│         Server Container            │
│  ┌───────────────────────────────┐  │
│  │   FastAPI Server (8000)       │  │
│  └───────────────────────────────┘  │
│              ↓                      │
│  ┌───────────────────────────────┐  │
│  │   LLM Inference               │  │
│  │   (Language Model Only)       │  │
│  │   GPU Accelerated             │  │
│  └───────────────────────────────┘  │
│              ↓                      │
│       generated_text                │
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
docker compose build

# Start services (server first, then client)
docker compose up

# Check logs
docker compose logs -f

# Access Gradio UI
# Open browser: http://localhost:7860

# Remove containers when done
docker compose down
```

### 2. Environment Configuration

Edit `docker-compose.yml` to customize:

```yaml
services:
  server:
    image: shjung-qwen3vl-server:v0.1      # Docker image name
    ports:
      - "8001:8001"                         # Server port mapping
    environment:
      - MODEL_NAME=Qwen/Qwen3-VL-2B-Instruct # Model to use
      - DEVICE_MAP=auto                      # Automatic device mapping
      - TORCH_DTYPE=auto                     # Automatic dtype selection
      - PORT=8001                            # Server port
      - CUDA_VISIBLE_DEVICES=0               # GPU visibility inside container
    volumes:
      - /mnt/ssd1/shjung/huggingface:/root/.cache/huggingface # Model cache
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['3']             # Host GPU 3 → Container GPU 0
              capabilities: [gpu]

  client:
    image: shjung-qwen3vl-client:v0.1      # Docker image name
    ports:
      - "7860:7860"                         # Gradio port mapping
    environment:
      - SERVER_URL=http://server:8001       # Server URL (internal network)
      - MODEL_NAME=Qwen/Qwen3-VL-2B-Instruct # Model to use
      - GRADIO_SERVER_NAME=0.0.0.0          # Gradio host (0.0.0.0 for external access)
      - GRADIO_SERVER_PORT=7860             # Gradio port
      - CUDA_VISIBLE_DEVICES=0              # GPU visibility inside container
    volumes:
      - /mnt/ssd1/shjung/huggingface:/root/.cache/huggingface # Model cache
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['2']             # Host GPU 2 → Container GPU 0
              capabilities: [gpu]
```

**GPU Allocation Notes:**
- Server uses host GPU 3, appears as `cuda:0` inside container
- Client uses host GPU 2, appears as `cuda:0` inside container
- `device_ids` sets physical GPU at Docker level
- `CUDA_VISIBLE_DEVICES=0` ensures app sees only that GPU

## 📁 Project Structure

```
.
├── docker-compose.yml          # Orchestration config
│
├── client/                     # Client container
│   ├── Dockerfile              # Client container build
│   ├── requirements.txt        # Python dependencies (frozen)
│   ├── preprocessor.py         # Vision preprocessing (qwen-vl-utils)
│   ├── vision_encoder.py       # Vision Encoder (2048-dim embeddings, CPU)
│   ├── client_api.py           # HTTP client for server communication
│   ├── gradio_app.py           # Gradio UI (port 7860)
│   ├── test_vision_embedding.py # Test vision encoder (images/videos)
│   └── save_test_embeddings.py # Save embeddings for debugging
│
├── server/                     # Server container
│   ├── Dockerfile              # Server container build
│   ├── requirements.txt        # Python dependencies (frozen)
│   ├── llm_inference.py        # LLM generation (Language Model on GPU)
│   ├── server_api.py           # FastAPI endpoints (port 8001)
│   └── test_server_with_embeddings.py # End-to-end test
│
├── qwen-vl-utils/              # Vision processing utilities (pip package)
│   └── src/qwen_vl_utils/
│       ├── __init__.py
│       └── vision_process.py   # Image/video loading, resizing
│
├── cookbooks/                  # Example notebooks
│   ├── video_understanding.ipynb
│   ├── ocr.ipynb
│   └── ...
│
└── README/                     # Documentation for Debugging and Development
    ├── README.md               # Implementation guide
    ├── CODE_MAPPING.md         # Source code mapping
    └── SERVER_TESTING_GUIDE.md # Testing guide
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
  "input_ids": [[1, 2, 3, ...]],              # Tokenized text
  "vision_embeddings": [[0.1, 0.2, ...], ...], # 2048-dim vision embeddings from client
  "vision_token_positions": [5, 10],          # Positions of <|vision_start|> tokens
  "attention_mask": [[1, 1, 1, ...]],         # Attention mask
  "max_new_tokens": 128,                       # Max tokens to generate
  "temperature": 0.7,                          # Sampling temperature
  "top_p": 0.8                                 # Nucleus sampling
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

Request: (same as /api/v1/generate)

Response: (newline-delimited JSON stream)
{"text": "Hello"}
{"text": " world"}
{"text": "!", "finish_reason": "stop"}
```

## 🛠️ Troubleshooting

### Issue: Package Version Error
```bash
# The Qwen3-VL model requires transformers >= 4.57.0
pip install "transformers>=4.57.0"

pip install qwen-vl-utils==0.0.14
# It's highly recommended to use `[decord]` feature for faster video loading.
# pip install qwen-vl-utils[decord]

pip install -U flash-attn --no-build-isolation
```

## 📝 Development Mode

Run without Docker for development:

### Client Setup
```bash
cd client

# Create virtual environment for client with uv 
uv venv -p 3.12
source .venv/bin/activate

# Install PyTorch with CUDA support
uv pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu128

# Install dependencies
uv pip install --no-build-isolation -r requirements.txt

# Test vision encoder
python test_vision_embedding.py image ../examples/dog.jpg
python test_vision_embedding.py video ../examples/IronMan.mp4

# Save test embeddings
python save_test_embeddings.py image ../examples/dog.jpg
python save_test_embeddings.py video ../examples/IronMan.mp4

# Run Gradio app
export SERVER_URL=http://localhost:8001
export HF_HOME="/mnt/ssd1/shjung/huggingface" # Huggingface cache directory
export CUDA_VISIBLE_DEVICES=2 # It should be different from server GPU
python gradio_app.py
```

**Important Dependencies:**
- `qwen-vl-utils`: Vision processing utilities
- `decord`: Video decoding (preferred backend, more stable than torchvision)
- `transformers>=4.57.0`: Qwen3-VL model support

### Server Setup
```bash
cd client

# Create virtual environment for client with uv 
uv venv -p 3.12
source .venv/bin/activate

# Install PyTorch & Flash-attention with CUDA support
uv pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu128
RUN uv pip install --system --no-build-isolation --no-cache-dir https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.4.22/flash_attn-2.8.1+cu128torch2.9-cp312-cp312-linux_x86_64.whl

# Install dependencies
uv pip install --no-build-isolation -r requirements.txt

# Test Server with saved test embeddings
python test_server_with_embeddings.py image ../pt_data_examples/image_dog_tensors.pt
python test_server_with_embeddings.py video ../pt_data_examples/video_ironman_tensors.pt

# Run Gradio app
export HF_HOME="/mnt/ssd1/shjung/huggingface" # Huggingface cache directory
export CUDA_VISIBLE_DEVICES=2 # It should be different from client GPU
python server_api.py
```

### Video Processing Notes

The client uses `qwen-vl-utils` for video processing with the following backends (in order of preference):
1. **decord** (recommended) - Most stable, install with `pip install decord`
2. torchcodec - Experimental
3. torchvision - Deprecated, avoid

If you encounter video processing errors, ensure `decord` is installed in your environment. and use ```pip install qwen-vl-utils[decord]``` to install the package with decord support.


## 📚 References

- [Qwen3-VL Model Card](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)
- [qwen-vl-utils Documentation](https://github.com/QwenLM/Qwen-VL)
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [Gradio Docs](https://gradio.app/)

## 📄 License

See LICENSE file in repository root.
