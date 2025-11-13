"""
FastAPI Server for LLM Inference
Receives preprocessed inputs from client and returns generated text
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import torch
import logging
import json
from llm_inference import ServerLLMInference
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Qwen3-VL Inference Server",
    description="LLM inference server for Qwen3-VL",
    version="1.0.0"
)

# Global model instance
llm_inference = None


# Request models
class GenerateRequest(BaseModel):
    """Request model for generation"""
    input_ids: List[List[int]]
    vision_embeddings: List[List[float]]
    vision_token_positions: List[int]
    attention_mask: List[List[int]]
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.8


class GenerateResponse(BaseModel):
    """Response model for generation"""
    text: str
    finish_reason: str = "stop"


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    global llm_inference
    
    logger.info("=" * 60)
    logger.info("🚀 Starting Qwen3-VL Inference Server")
    logger.info("=" * 60)
    
    # Get model config from environment
    model_name = os.getenv("MODEL_NAME", "Qwen/Qwen3-VL-2B-Instruct")
    device_map = os.getenv("DEVICE_MAP", "auto")
    torch_dtype = os.getenv("TORCH_DTYPE", "auto")
    
    logger.info(f"Model: {model_name}")
    logger.info(f"Device: {device_map}")
    
    # Initialize inference engine
    llm_inference = ServerLLMInference(
        model_name=model_name,
        device_map=device_map,
        torch_dtype=torch_dtype
    )
    
    logger.info("✅ Server ready!")
    logger.info("=" * 60)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if llm_inference is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "device": str(llm_inference.device)
    }


@app.post("/api/v1/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """
    Generate text from preprocessed inputs (non-streaming)
    
    Args:
        request: GenerateRequest with input_ids, vision_embeddings, etc.
        
    Returns:
        GenerateResponse with generated text
    """
    if llm_inference is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        logger.info("📥 Received generation request")
        logger.info(f"   Input IDs shape: {len(request.input_ids)}x{len(request.input_ids[0])}")
        logger.info(f"   Vision embeddings: {len(request.vision_embeddings)}x{len(request.vision_embeddings[0])}")
        
        # Convert lists to tensors
        input_ids = torch.tensor(request.input_ids, dtype=torch.long)
        vision_embeddings = torch.tensor(request.vision_embeddings, dtype=torch.float32)
        attention_mask = torch.tensor(request.attention_mask, dtype=torch.long)
        
        # Generate
        generated_text = llm_inference.generate(
            input_ids=input_ids,
            vision_embeddings=vision_embeddings,
            vision_token_positions=request.vision_token_positions,
            attention_mask=attention_mask,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )
        
        logger.info(f"✅ Generated {len(generated_text)} characters")
        
        return GenerateResponse(
            text=generated_text,
            finish_reason="stop"
        )
        
    except Exception as e:
        logger.error(f"❌ Generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/generate_stream")
async def generate_stream(request: GenerateRequest):
    """
    Generate text with streaming response
    
    Args:
        request: GenerateRequest with input_ids, vision_embeddings, etc.
        
    Returns:
        StreamingResponse with JSON lines
    """
    if llm_inference is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    async def stream_generator():
        """Generator for streaming response"""
        try:
            logger.info("📥 Received streaming generation request")
            
            # Convert lists to tensors
            input_ids = torch.tensor(request.input_ids, dtype=torch.long)
            vision_embeddings = torch.tensor(request.vision_embeddings, dtype=torch.float32)
            attention_mask = torch.tensor(request.attention_mask, dtype=torch.long)
            
            # Stream generation
            for result in llm_inference.generate_stream(
                input_ids=input_ids,
                vision_embeddings=vision_embeddings,
                vision_token_positions=request.vision_token_positions,
                attention_mask=attention_mask,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_p=request.top_p
            ):
                # Yield JSON line
                yield json.dumps(result).encode('utf-8') + b'\n'
                
        except Exception as e:
            logger.error(f"❌ Streaming failed: {e}", exc_info=True)
            error_response = {"error": str(e)}
            yield json.dumps(error_response).encode('utf-8') + b'\n'
    
    return StreamingResponse(
        stream_generator(),
        media_type="application/x-ndjson"  # Newline-delimited JSON
    )


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Qwen3-VL Inference Server",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "generate": "/api/v1/generate",
            "generate_stream": "/api/v1/generate_stream"
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment
    port = int(os.getenv("PORT", "8000"))
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
