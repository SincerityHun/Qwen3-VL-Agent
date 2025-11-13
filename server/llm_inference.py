"""
Server-side LLM Inference Module
Handles generation using only the Language Model (skips Vision Encoder)
"""

from transformers import Qwen3VLForConditionalGeneration, AutoTokenizer
import torch
from typing import Optional, Iterator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ServerLLMInference:
    """
    LLM Inference server that accepts vision embeddings from client
    Only runs the Language Model part (Prefill + Decoding)
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-2B-Instruct",
        device_map: str = "auto",
        torch_dtype: str = "auto"
    ):
        """
        Initialize LLM for inference
        
        Args:
            model_name: HuggingFace model identifier
            device_map: Device mapping strategy
            torch_dtype: Data type for model weights
        """
        logger.info("=" * 60)
        logger.info("🚀 Initializing Server LLM Inference")
        logger.info("=" * 60)
        logger.info(f"Model: {model_name}")
        logger.info(f"Device map: {device_map}")
        
        # Load full model (but we'll only use language_model part)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            attn_implementation="flash_attention_2"  # Use FlashAttention for speed
        )
        self.model.eval()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Get model device
        self.device = next(self.model.parameters()).device
        
        # Token IDs
        self.IMAGE_TOKEN_ID = 151655
        self.VIDEO_TOKEN_ID = 151656
        
        logger.info(f"✅ Model loaded on {self.device}")
        logger.info(f"   Model type: {type(self.model).__name__}")
        logger.info(f"   Parameters: {sum(p.numel() for p in self.model.parameters()) / 1e9:.2f}B")
        logger.info("=" * 60)
    
    def _merge_vision_embeddings(
        self,
        input_ids: torch.Tensor,
        text_embeds: torch.Tensor,
        vision_embeddings: torch.Tensor,
        vision_token_positions: list
    ) -> torch.Tensor:
        """
        Merge vision embeddings into text embeddings at specified positions
        
        Args:
            input_ids: Token IDs [batch, seq_len]
            text_embeds: Text embeddings [batch, seq_len, hidden_dim]
            vision_embeddings: Vision embeddings [num_patches, hidden_dim]
            vision_token_positions: Positions where vision tokens appear
            
        Returns:
            Merged embeddings [batch, seq_len, hidden_dim]
        """
        batch_size, seq_len, hidden_dim = text_embeds.shape
        inputs_embeds = text_embeds.clone()
        
        vision_idx = 0
        for batch_idx in range(batch_size):
            for pos in vision_token_positions:
                token_id = input_ids[batch_idx, pos].item()
                
                # Replace vision token with vision embedding
                if token_id in [self.IMAGE_TOKEN_ID, self.VIDEO_TOKEN_ID]:
                    if vision_idx < vision_embeddings.shape[0]:
                        inputs_embeds[batch_idx, pos] = vision_embeddings[vision_idx]
                        vision_idx += 1
        
        logger.info(f"✅ Merged {vision_idx} vision embeddings")
        return inputs_embeds
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        vision_embeddings: torch.Tensor,
        vision_token_positions: list,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.8,
        do_sample: bool = True
    ) -> str:
        """
        Generate text from preprocessed inputs (non-streaming)
        
        Args:
            input_ids: Token IDs [batch, seq_len]
            vision_embeddings: Vision features [num_patches, hidden_dim]
            vision_token_positions: Positions of vision tokens
            attention_mask: Attention mask [batch, seq_len]
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling
            
        Returns:
            Generated text string
        """
        logger.info("🔥 Starting generation...")
        logger.info(f"   Input shape: {input_ids.shape}")
        logger.info(f"   Vision embeddings: {vision_embeddings.shape}")
        logger.info(f"   Vision positions: {vision_token_positions}")
        
        # Move inputs to device
        input_ids = input_ids.to(self.device)
        vision_embeddings = vision_embeddings.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        # 1. Get text embeddings
        text_embeds = self.model.language_model.embed_tokens(input_ids)
        logger.info(f"   Text embeddings: {text_embeds.shape}")
        
        # 2. Merge vision embeddings
        inputs_embeds = self._merge_vision_embeddings(
            input_ids, text_embeds, vision_embeddings, vision_token_positions
        )
        
        # 3. Generate with LLM
        logger.info("🚀 Running LLM generation...")
        generated_ids = self.model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            use_cache=True,  # Use KV cache for efficiency
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        # 4. Decode only the new tokens
        new_tokens = generated_ids[0, input_ids.shape[1]:]
        generated_text = self.tokenizer.decode(
            new_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        logger.info(f"✅ Generation complete: {len(generated_text)} chars")
        return generated_text
    
    @torch.no_grad()
    def generate_stream(
        self,
        input_ids: torch.Tensor,
        vision_embeddings: torch.Tensor,
        vision_token_positions: list,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.8
    ) -> Iterator[dict]:
        """
        Generate text with streaming (yields partial results)
        
        Yields:
            Dictionary with 'text' and optionally 'finish_reason'
        """
        logger.info("🔥 Starting streaming generation...")
        
        # Move inputs to device
        input_ids = input_ids.to(self.device)
        vision_embeddings = vision_embeddings.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        # Get merged embeddings
        text_embeds = self.model.language_model.embed_tokens(input_ids)
        inputs_embeds = self._merge_vision_embeddings(
            input_ids, text_embeds, vision_embeddings, vision_token_positions
        )
        
        # Streaming generation
        from transformers import TextIteratorStreamer
        from threading import Thread
        
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )
        
        generation_kwargs = {
            'inputs_embeds': inputs_embeds,
            'attention_mask': attention_mask,
            'max_new_tokens': max_new_tokens,
            'temperature': temperature,
            'top_p': top_p,
            'do_sample': True,
            'streamer': streamer,
            'use_cache': True,
            'pad_token_id': self.tokenizer.pad_token_id,
            'eos_token_id': self.tokenizer.eos_token_id
        }
        
        # Run generation in background thread
        thread = Thread(
            target=self.model.language_model.generate,
            kwargs=generation_kwargs
        )
        thread.start()
        
        # Yield generated tokens
        accumulated_text = ""
        for new_text in streamer:
            accumulated_text += new_text
            yield {"text": accumulated_text}
        
        yield {"text": accumulated_text, "finish_reason": "stop"}
        logger.info(f"✅ Streaming complete: {len(accumulated_text)} chars")
