# Server Dockerfile for Qwen3-VL Inference
ARG CUDA_VERSION=12.8.1
ARG PYTHON_VERSION=3.12

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04 AS base

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# Set working directory
WORKDIR /app

# Install Python 3.12
RUN apt-get update -y && apt-get install -y \
    software-properties-common curl wget git sudo build-essential vim \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update -y \
    && apt-get install -y python3.12 python3.12-dev python3.12-venv \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 \
    && python3 --version && pip3 --version

# Create symbolic link for python
RUN ln -s /usr/bin/python3.12 /usr/bin/python

# Copy requirements
RUN python3 -m pip install -U pip uv
ENV UV_HTTP_TIMEOUT=500

# Copy Project package (Torch, flash-attention for CUDA 12.8)
RUN uv pip install --system --no-build-isolation --no-cache-dir torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu128
RUN uv pip install --system --no-build-isolation --no-cache-dir https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.4.22/flash_attn-2.8.1+cu128torch2.9-cp312-cp312-linux_x86_64.whl


# Install Python dependencies
COPY requirements.txt .
RUN uv pip install --system --no-build-isolation --no-cache-dir -r requirements.txt

# Copy application code
COPY llm_inference.py .
COPY server_api.py .

# Expose FastAPI port
EXPOSE 8001

# Set environment variables
ENV MODEL_NAME=Qwen/Qwen3-VL-2B-Instruct
ENV DEVICE_MAP=auto
ENV TORCH_DTYPE=auto
ENV PORT=8001

# Run FastAPI server
CMD ["python", "server_api.py"]
