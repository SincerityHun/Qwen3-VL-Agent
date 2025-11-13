"""
Client API for sending preprocessed data to server
Handles communication with FastAPI server
"""

import requests
import json
import torch
import numpy as np
from typing import Dict, List, Any, Iterator
import base64
import io


class ClientAPI:
    """
    Client API for communicating with inference server
    Sends preprocessed inputs and receives generated text
    """
    
    def __init__(self, server_url: str = "http://localhost:8000"):
        """
        Initialize client API
        
        Args:
            server_url: Base URL of the FastAPI server
        """
        self.server_url = server_url
        self.health_endpoint = f"{server_url}/health"
        self.generate_endpoint = f"{server_url}/api/v1/generate"
        self.generate_stream_endpoint = f"{server_url}/api/v1/generate_stream"
        
    def check_health(self) -> bool:
        """Check if server is healthy"""
        try:
            response = requests.get(self.health_endpoint, timeout=5)
            return response.status_code == 200
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return False
    
    def _tensor_to_list(self, tensor: torch.Tensor) -> List:
        """Convert tensor to nested list for JSON serialization"""
        return tensor.cpu().numpy().tolist()
    
    def _prepare_payload(
        self,
        input_ids: torch.Tensor,
        vision_embeddings: torch.Tensor,
        vision_token_positions: List[int],
        attention_mask: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.8
    ) -> Dict[str, Any]:
        """
        Prepare request payload for server
        
        Args:
            input_ids: Token IDs [batch, seq_len]
            vision_embeddings: Vision features [num_patches, hidden_dim]
            vision_token_positions: Positions of vision tokens
            attention_mask: Attention mask [batch, seq_len]
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            
        Returns:
            Dictionary ready for JSON serialization
        """
        payload = {
            "input_ids": self._tensor_to_list(input_ids),
            "vision_embeddings": self._tensor_to_list(vision_embeddings),
            "vision_token_positions": vision_token_positions,
            "attention_mask": self._tensor_to_list(attention_mask),
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p
        }
        
        return payload
    
    def generate(
        self,
        input_ids: torch.Tensor,
        vision_embeddings: torch.Tensor,
        vision_token_positions: List[int],
        attention_mask: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.8
    ) -> str:
        """
        Generate text (non-streaming)
        
        Returns:
            Generated text string
        """
        print("📡 Sending request to server...")
        
        payload = self._prepare_payload(
            input_ids, vision_embeddings, vision_token_positions,
            attention_mask, max_new_tokens, temperature, top_p
        )
        
        # Calculate payload size
        payload_size = len(json.dumps(payload)) / 1024 / 1024  # MB
        print(f"   Payload size: {payload_size:.2f} MB")
        
        try:
            response = requests.post(
                self.generate_endpoint,
                json=payload,
                timeout=300  # 5 minutes timeout
            )
            response.raise_for_status()
            
            result = response.json()
            generated_text = result.get('text', '')
            
            print(f"✅ Received response: {len(generated_text)} chars")
            return generated_text
            
        except Exception as e:
            print(f"❌ Request failed: {e}")
            raise
    
    def generate_stream(
        self,
        input_ids: torch.Tensor,
        vision_embeddings: torch.Tensor,
        vision_token_positions: List[int],
        attention_mask: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.8
    ) -> Iterator[str]:
        """
        Generate text (streaming)
        
        Yields:
            Partial generated text strings
        """
        print("📡 Streaming request to server...")
        
        payload = self._prepare_payload(
            input_ids, vision_embeddings, vision_token_positions,
            attention_mask, max_new_tokens, temperature, top_p
        )
        
        try:
            response = requests.post(
                self.generate_stream_endpoint,
                json=payload,
                stream=True,
                timeout=300
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode('utf-8'))
                        if 'text' in data:
                            yield data['text']
                        if data.get('finish_reason'):
                            break
                    except json.JSONDecodeError:
                        continue
                        
        except Exception as e:
            print(f"❌ Streaming failed: {e}")
            raise
