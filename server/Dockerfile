# Server Dockerfile for Qwen3-VL Inference

ARG CUDA_VERSION=12.8.1
ARG PYTHON_VERSION=3.12

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04 AS base
# Set working directory
WORKDIR /app

# Install Python 3.12
RUN apt-get update -y && apt-get install -y \
    software-properties-common curl wget git sudo build-essential vim \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update -y \
    && apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python${PYTHON_VERSION} \
    && python3 --version && pip3 --version

# Create symbolic link for python
RUN ln -s /usr/bin/python3.${PYTHON_VERSION} /usr/bin/python

# Copy requirements
RUN python3 -m pip install -U pip uv
ENV UV_HTTP_TIMEOUT=500

RUN uv pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu128
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

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
