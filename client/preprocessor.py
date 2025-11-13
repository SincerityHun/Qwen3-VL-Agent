"""
Client-side preprocessor for Qwen3-VL
Handles image/video loading, resizing, and tokenization
"""

import sys
import os
# sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'qwen-vl-utils', 'src'))

from qwen_vl_utils import (
    extract_vision_info,
    fetch_image,
    fetch_video,
    process_vision_info,
    smart_resize
)
from transformers import AutoProcessor
import torch
from typing import Dict, List, Any, Optional


class ClientPreprocessor:
    """
    Client-side preprocessor for multimodal inputs
    Performs vision preprocessing and tokenization on-device
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen3-VL-2B-Instruct"):
        """
        Initialize preprocessor with model-specific processor
        
        Args:
            model_name: HuggingFace model identifier
        """
        print(f"📦 Loading processor for {model_name}...")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.image_patch_size = self.processor.image_processor.patch_size
        print(f"✅ Processor loaded! Patch size: {self.image_patch_size}")
        
    def preprocess(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Preprocess messages containing text, images, and videos
        
        Args:
            messages: List of message dicts with role and content
            
        Returns:
            Dictionary containing preprocessed inputs:
            - input_ids: Token IDs
            - attention_mask: Attention mask
            - pixel_values: Image/video pixels (if any)
            - image_grid_thw: Image grid dimensions
            - video_grid_thw: Video grid dimensions
        """
        print("🔄 Preprocessing vision information...")
        
        # 1. Apply chat template
        text = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        print(f"📝 Text template applied: {len(text)} chars")
        
        # 2. Process vision information (images/videos)
        # Note: process_vision_info expects a LIST of conversations
        # Each conversation is a list of messages
        images, videos, video_kwargs = process_vision_info(
            [messages],  # Wrap messages in a list!
            image_patch_size=self.image_patch_size,
            return_video_kwargs=True,
            return_video_metadata=True
        )
        
        # Debug: print what we got
        # print(f"🔍 Debug - images: {type(images)}, {images is not None and len(images) if images else 0}")
        # print(f"🔍 Debug - videos: {type(videos)}, {videos is not None and len(videos) if videos else 0}")
        # print(f"� Debug - video_kwargs: {video_kwargs}")
        
        # if images:
        #     print(f"�🖼️  Processed {len(images)} images")
        # if videos:
        #     print(f"🎬 Processed {len(videos)} videos")
        #     print(f"   Video type: {type(videos[0]) if videos else None}")
        #     if videos and len(videos) > 0:
        #         if isinstance(videos[0], tuple):
        #             print(f"   Video[0] is tuple with {len(videos[0])} elements")
        #             print(f"   Video tensor shape: {videos[0][0].shape if hasattr(videos[0][0], 'shape') else 'N/A'}")

        
        # 3. Handle video metadata (for Qwen3VL)
        video_metadatas = None
        if videos is not None:
            videos, video_metadatas = zip(*videos)
            videos, video_metadatas = list(videos), list(video_metadatas)
            # print(f"🔍 Debug - After unzipping:")
            # print(f"   videos type: {type(videos)}, len: {len(videos)}")
            # print(f"   videos[0] type: {type(videos[0])}, shape: {videos[0].shape if hasattr(videos[0], 'shape') else 'N/A'}")
            # print(f"   video_metadatas type: {type(video_metadatas)}, len: {len(video_metadatas)}")

        
        # 4. Tokenize and create model inputs
        # Note: processor expects text as a list for batch processing
        # print(f"🔍 Debug - Before processor:")
        # print(f"   text type: {type([text])}, len: {len([text])}")
        # print(f"   images: {images}")
        # print(f"   videos type: {type(videos)}, content: {videos if videos and len(videos) < 3 else 'too long'}")
        # print(f"   video_metadata type: {type(video_metadatas)}")
        # print(f"   video_kwargs: {video_kwargs}")
        
        inputs = self.processor(
            text=[text],  # Text must be a list!
            images=images,
            videos=videos,
            video_metadata=video_metadatas,
            do_resize=False,  # Already resized by process_vision_info
            return_tensors="pt",
            **video_kwargs
        )
        
        # print(f"🔍 Debug - After processor, keys: {inputs.keys()}")
        
        print(f"✅ Preprocessing complete!")
        print(f"   Input IDs shape: {inputs['input_ids'].shape}")
        if 'pixel_values' in inputs:
            print(f"   Pixel values (images) shape: {inputs['pixel_values'].shape}")
        if 'pixel_values_videos' in inputs:
            print(f"   Pixel values (videos) shape: {inputs['pixel_values_videos'].shape}")
        
        return inputs
    
    def extract_vision_token_positions(self, input_ids: torch.Tensor) -> List[int]:
        """
        Find positions of vision tokens (<image>, <video>) in input_ids
        
        Args:
            input_ids: Token ID tensor [batch, seq_len]
            
        Returns:
            List of positions where vision tokens appear
        """
        IMAGE_TOKEN_ID = 151655
        VIDEO_TOKEN_ID = 151656
        
        positions = []
        for i, token_id in enumerate(input_ids[0].tolist()):
            if token_id in [IMAGE_TOKEN_ID, VIDEO_TOKEN_ID]:
                positions.append(i)
        
        return positions
