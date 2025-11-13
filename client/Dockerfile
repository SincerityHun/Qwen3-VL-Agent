# Client Dockerfile for Qwen3-VL Inference
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
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

# Run Gradio app
CMD ["python", "gradio_app.py"]
