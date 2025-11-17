# Server Dockerfile for Qwen3-VL Inference
ARG CUDA_VERSION=12.8.1
ARG PYTHON_VERSION=3.12

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04 AS base

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# Set working directory
WORKDIR /app

# Install system dependencies
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

# Install PyTorch for CUDA 12.8 (before requirements.txt)
RUN uv pip install --system --no-build-isolation --no-cache-dir torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu128

# Copy and install other dependencies (skip torch dependencies)
COPY requirements.txt .
RUN uv pip install --system --no-build-isolation --no-cache-dir -r requirements.txt

# Copy application code
COPY preprocessor.py .
COPY vision_encoder.py .
COPY client_api.py .
COPY gradio_app.py .

# Expose Gradio port
EXPOSE 7860

# Set environment variables
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860
ENV SERVER_URL=http://server:8001
ENV USE_VISION_ENCODER=true
ENV MODEL_NAME=Qwen/Qwen3-VL-2B-Instruct

# Run Gradio app
CMD ["python", "gradio_app.py"]
