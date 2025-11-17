"""
Server-side LLM Inference Module
Handles generation using only the Language Model (skips Vision Encoder)

Architecture:
- Vision Encoder: Kept on CPU (not used, saves GPU memory)
- Language Model: Loaded on GPU (performs actual inference)
- Vision embeddings: Received from client, merged into text embeddings

Critical Design Decisions:
1. CPU Offloading: Vision Encoder stays on CPU to save ~0.6GB GPU memory
2. Device Management: model.device property is patched during generation to return
   Language Model's device (cuda:0) instead of Vision Encoder's device (cpu)
3. Token ID Handling: All special token IDs stored as Python int (not tensors)
   to avoid device mismatch errors during generation
4. Dual Input: Both input_ids and inputs_embeds passed to generate() for proper
   sequence length tracking while using custom embeddings
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
        
        # Load full model on CPU first (to avoid loading Vision Encoder on GPU)
        logger.info("📦 Loading full model on CPU (will extract Language Model only)...")
        self.full_model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="cpu"  # ← Load on CPU first!
        )
        
        # Determine target device for Language Model
        if device_map == "auto":
            target_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            target_device = device_map
        
        logger.info(f"🎯 Moving Language Model to {target_device}...")
        
        # Move only Language Model to GPU
        self.full_model.model.language_model = self.full_model.model.language_model.to(target_device)
        self.full_model.lm_head = self.full_model.lm_head.to(target_device)
        
        # Vision Encoder stays on CPU (we don't use it on server)
        logger.info("📦 Vision Encoder kept on CPU (not used on server)...")
        
        self.full_model.eval()
        
        # Keep only language model reference for convenience
        self.model = self.full_model
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Get model device
        self.device = next(self.full_model.model.language_model.parameters()).device
        
        # Token IDs (store as Python int to avoid device mismatch issues)
        self.IMAGE_TOKEN_ID = 151655
        self.VIDEO_TOKEN_ID = 151656
        self.pad_token_id = int(self.tokenizer.pad_token_id) if self.tokenizer.pad_token_id is not None else 151643
        self.eos_token_id = int(self.tokenizer.eos_token_id) if self.tokenizer.eos_token_id is not None else 151645
        
        # CRITICAL FIX: Update model configs with Python int to prevent device mismatch
        # When Vision Encoder is on CPU but Language Model is on GPU,
        # config token IDs must be Python int (not tensors) to avoid device conflicts
        if hasattr(self.full_model, 'config'):
            self.full_model.config.pad_token_id = self.pad_token_id
            self.full_model.config.eos_token_id = self.eos_token_id
            
        if hasattr(self.full_model, 'generation_config'):
            self.full_model.generation_config.pad_token_id = self.pad_token_id
            self.full_model.generation_config.eos_token_id = self.eos_token_id
        
        # Calculate actual GPU memory usage
        language_params = sum(p.numel() for p in self.full_model.model.language_model.parameters()) / 1e9
        visual_params = sum(p.numel() for p in self.full_model.visual.parameters()) / 1e9
        
        # Check device allocation
        visual_device = next(self.full_model.visual.parameters()).device
        language_device = next(self.full_model.model.language_model.parameters()).device
        
        logger.info(f"✅ Language Model loaded!")
        logger.info(f"   Language Model: {language_params:.2f}B params on {language_device}")
        logger.info(f"   Vision Encoder: {visual_params:.2f}B params on {visual_device} (not used)")
        logger.info(f"   Total params: {sum(p.numel() for p in self.full_model.parameters()) / 1e9:.2f}B")
        logger.info(f"   GPU memory: Only Language Model (~{language_params:.2f}B params)")
        logger.info(f"   Token IDs: pad={self.pad_token_id}, eos={self.eos_token_id}")
        
        # Verify Vision Encoder is on CPU
        if visual_device.type == 'cpu':
            logger.info(f"   ✅ Vision Encoder on CPU (memory efficient)")
        else:
            logger.warning(f"   ⚠️  Vision Encoder is on {visual_device}, not CPU!")
        
        # Clean up GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
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
        text_embeds = self.model.model.language_model.embed_tokens(input_ids)
        logger.info(f"   Text embeddings: {text_embeds.shape}")
        
        # 2. Merge vision embeddings into text embeddings
        inputs_embeds = self._merge_vision_embeddings(
            input_ids, text_embeds, vision_embeddings, vision_token_positions
        )
        
        # 3. CRITICAL FIX: Temporarily patch model.device property
        # When Vision Encoder is on CPU, model.device returns 'cpu' (first parameter device)
        # But we need it to return cuda:0 (Language Model device) for correct tensor creation
        original_device_fn = None
        if hasattr(self.model, 'device'):
            original_device_fn = type(self.model).device
            # Monkey patch to return Language Model's device instead of Vision Encoder's
            type(self.model).device = property(lambda self: next(self.model.language_model.parameters()).device)
        
        try:
            # 4. Generate with Language Model
            # Note: When both input_ids and inputs_embeds are provided:
            # - input_ids helps determine sequence length and position embeddings
            # - inputs_embeds is actually used (Vision Encoder is skipped)
            logger.info("🚀 Running LLM generation...")
            
            generated_ids = self.model.generate(
                input_ids=input_ids,  # For sequence length tracking
                inputs_embeds=inputs_embeds,  # Actual embeddings to use
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                use_cache=True,
                pad_token_id=self.pad_token_id,
                eos_token_id=self.eos_token_id
            )
            
        finally:
            # Restore original device property
            if original_device_fn is not None:
                type(self.model).device = original_device_fn
        
        # 5. Decode only the new tokens
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
        text_embeds = self.model.model.language_model.embed_tokens(input_ids)
        inputs_embeds = self._merge_vision_embeddings(
            input_ids, text_embeds, vision_embeddings, vision_token_positions
        )
        
        # Setup streaming
        from transformers import TextIteratorStreamer
        from threading import Thread
        
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )
        
        generation_kwargs = {
            'input_ids': input_ids,  # For sequence length tracking
            'inputs_embeds': inputs_embeds,  # Actual embeddings to use
            'attention_mask': attention_mask,
            'max_new_tokens': max_new_tokens,
            'temperature': temperature,
            'top_p': top_p,
            'do_sample': True,
            'streamer': streamer,
            'use_cache': True,
            'pad_token_id': self.pad_token_id,
            'eos_token_id': self.eos_token_id
        }
        
        # Run generation in background thread
        thread = Thread(
            target=self.model.generate,
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
