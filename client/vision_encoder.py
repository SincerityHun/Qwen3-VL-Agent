"""
Client-side Vision Encoder
Extracts and runs only the Vision Encoder (ViT + DeepStack) from Qwen3-VL
"""

from transformers import Qwen3VLForConditionalGeneration
import torch
import os
from typing import Optional


class ClientVisionEncoder:
    """
    Vision Encoder for client-side processing
    Extracts visual features from images/videos using ViT + DeepStack
    """
    
    def __init__(
        self, 
        model_name: str = "Qwen/Qwen3-VL-2B-Instruct",
        device: Optional[str] = None
    ):
        """
        Initialize Vision Encoder
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run encoder on (e.g., 'cuda:0', 'cuda:1', 'cpu')
                   If None, uses CUDA_VISIBLE_DEVICES env var or default cuda
        """
        # Determine device
        if device is None:
            if torch.cuda.is_available():
                # CUDA_VISIBLE_DEVICES remaps GPUs, so always use cuda:0
                # Example: CUDA_VISIBLE_DEVICES=3 makes GPU 3 appear as cuda:0
                device = 'cuda:0'
            else:
                device = 'cpu'
        
        print(f"🚀 Loading Vision Encoder from {model_name}...")
        print(f"   Target device: {device}")
        print(f"   CUDA_VISIBLE_DEVICES: {os.getenv('CUDA_VISIBLE_DEVICES', 'not set')}")
        if torch.cuda.is_available():
            print(f"   Available GPUs: {torch.cuda.device_count()}")
            print(f"   GPU 0 name: {torch.cuda.get_device_name(0)}")
        
        # Load full model on CPU first (to avoid loading Language Model on GPU)
        print(f"📦 Loading full model on CPU (will extract Vision Encoder only)...")
        full_model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="cpu"  # ← Load on CPU first!
        )
        
        # Extract Vision Encoder and move to target device
        print(f"🎯 Extracting Vision Encoder and moving to {device}...")
        self.vision_encoder = full_model.visual.to(device)
        self.vision_encoder.eval()
        self.device = device
        
        # Delete Language Model immediately (before it touches GPU)
        print(f"🗑️  Deleting Language Model (not used on client)...")
        del full_model.model
        del full_model.lm_head
        del full_model
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Calculate memory usage
        vision_params = sum(p.numel() for p in self.vision_encoder.parameters()) / 1e9
        actual_device = next(self.vision_encoder.parameters()).device
        
        print(f"✅ Vision Encoder loaded!")
        print(f"   Model: {type(self.vision_encoder).__name__}")
        print(f"   Parameters: {vision_params:.2f}B")
        print(f"   Device: {actual_device}")
        
    @torch.no_grad()
    def encode(
        self, 
        pixel_values: torch.Tensor,
        image_grid_thw: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode images/videos to vision embeddings
        
        Args:
            pixel_values: Pixel values tensor
                - For images: [num_patches, hidden_dim] 
                - For videos: [num_patches, hidden_dim]
            image_grid_thw: Image grid dimensions [num_images, 3] (T, H, W)
            video_grid_thw: Video grid dimensions [num_videos, 3] (T, H, W)
            
        Returns:
            Vision embeddings [batch, num_patches, hidden_dim]
        """
        print("🎨 Encoding vision features...")
        print(f"   Input pixel_values shape: {pixel_values.shape}")
        
        # Combine image and video grids
        if image_grid_thw is not None and video_grid_thw is not None:
            grid_thw = torch.cat([image_grid_thw, video_grid_thw], dim=0)
        elif image_grid_thw is not None:
            grid_thw = image_grid_thw
        elif video_grid_thw is not None:
            grid_thw = video_grid_thw
        else:
            raise ValueError("Either image_grid_thw or video_grid_thw must be provided")
        
        print(f"   Grid THW shape: {grid_thw.shape}, values: {grid_thw}")
        
        # Run Vision Encoder
        # This includes: Patch Embedding → ViT Blocks → DeepStack Merger
        # Note: The model expects 'hidden_states' not 'pixel_values'
        vision_outputs = self.vision_encoder(
            hidden_states=pixel_values.to(self.device),
            grid_thw=grid_thw.to(self.device)
        )
        
        # Vision encoder may return tuple (hidden_states, ...) or just tensor
        if isinstance(vision_outputs, tuple):
            vision_embeddings = vision_outputs[0]
            print(f"✅ Vision encoding complete!")
            print(f"   Output shape: {vision_embeddings.shape}")
            print(f"   Note: Vision encoder returned {len(vision_outputs)} outputs (using first)")
        else:
            vision_embeddings = vision_outputs
            print(f"✅ Vision encoding complete!")
            print(f"   Output shape: {vision_embeddings.shape}")
        
        return vision_embeddings
    
    def get_embedding_size(self) -> int:
        """Get the hidden dimension size of vision embeddings"""
        # Access the config to get hidden size
        return self.vision_encoder.config.hidden_size if hasattr(self.vision_encoder, 'config') else 3584
