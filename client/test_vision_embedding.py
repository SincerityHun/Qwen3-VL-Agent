"""
Test script for vision embedding generation
Tests preprocessor and vision encoder without requiring server
"""

import torch
from preprocessor import ClientPreprocessor
from vision_encoder import ClientVisionEncoder
import sys


def test_vision_embedding(video_path: str, model_name: str = "Qwen/Qwen3-VL-2B-Instruct"):
    """
    Test vision embedding generation for a video input
    
    Args:
        video_path: Path to video file
        model_name: Model to use for processing
    """
    print("=" * 80)
    print("🧪 Testing Vision Embedding Generation")
    print("=" * 80)
    
    # 1. Initialize components
    print("\n[Step 1/4] Initializing Preprocessor...")
    preprocessor = ClientPreprocessor(model_name)
    
    print("\n[Step 2/4] Initializing Vision Encoder...")
    vision_encoder = ClientVisionEncoder(model_name)
    
    # 2. Create test message with video
    print(f"\n[Step 3/4] Processing video: {video_path}")
    # Use absolute path without file:// prefix (as shown in cookbooks)
    import os
    video_abs_path = os.path.abspath(video_path)
    messages = [
        {
            "role": "user",
            "content": [
                {"video": video_abs_path},  # Direct path, no "type" or "file://" prefix
                {"type": "text", "text": "Describe what happens in this video."}
            ]
        }
    ]
    
    # 3. Preprocess
    print("\n   📦 Preprocessing...")
    inputs = preprocessor.preprocess(messages)
    
    print("\n   ✅ Preprocessing Results:")
    print(f"      - input_ids shape: {inputs['input_ids'].shape}")
    print(f"      - attention_mask shape: {inputs['attention_mask'].shape}")
    
    if 'pixel_values' in inputs:
        print(f"      - pixel_values shape: {inputs['pixel_values'].shape}")
        print(f"      - pixel_values dtype: {inputs['pixel_values'].dtype}")
        print(f"      - pixel_values device: {inputs['pixel_values'].device}")
    
    if 'pixel_values_videos' in inputs:
        print(f"      - pixel_values_videos shape: {inputs['pixel_values_videos'].shape}")
        print(f"      - pixel_values_videos dtype: {inputs['pixel_values_videos'].dtype}")
        print(f"      - pixel_values_videos device: {inputs['pixel_values_videos'].device}")
    
    if 'image_grid_thw' in inputs:
        print(f"      - image_grid_thw: {inputs['image_grid_thw']}")
    
    if 'video_grid_thw' in inputs:
        print(f"      - video_grid_thw shape: {inputs['video_grid_thw'].shape}")
        print(f"      - video_grid_thw values: {inputs['video_grid_thw']}")
    
    # 4. Encode vision
    print("\n[Step 4/4] Generating Vision Embeddings...")
    
    # Check for either pixel_values (images) or pixel_values_videos (videos)
    if 'pixel_values_videos' in inputs:
        pixel_values = inputs['pixel_values_videos']
    elif 'pixel_values' in inputs:
        pixel_values = inputs['pixel_values']
    else:
        print("   ⚠️  No pixel_values found - video might not have been processed")
        return False
    
    vision_embeddings = vision_encoder.encode(
        pixel_values,
        inputs.get('image_grid_thw'),
        inputs.get('video_grid_thw')
    )
    
    print("\n   ✅ Vision Embedding Results:")
    print(f"      - Shape: {vision_embeddings.shape}")
    print(f"      - Dtype: {vision_embeddings.dtype}")
    print(f"      - Device: {vision_embeddings.device}")
    print(f"      - Min value: {vision_embeddings.min().item():.4f}")
    print(f"      - Max value: {vision_embeddings.max().item():.4f}")
    print(f"      - Mean value: {vision_embeddings.mean().item():.4f}")
    print(f"      - Std value: {vision_embeddings.std().item():.4f}")
    
    # Determine dimensions
    if vision_embeddings.ndim == 3:
        batch_size, seq_len, hidden_dim = vision_embeddings.shape
    elif vision_embeddings.ndim == 2:
        seq_len, hidden_dim = vision_embeddings.shape
        batch_size = 1
    else:
        print(f"      ⚠️  Unexpected embedding dimensions: {vision_embeddings.ndim}")
        hidden_dim = None
    
    if hidden_dim:
        print(f"      - Parsed: batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}")
    
    # 5. Get vision token positions
    vision_positions = preprocessor.extract_vision_token_positions(inputs['input_ids'])
    print(f"\n   🎯 Vision Token Positions: {vision_positions}")
    
    # 6. Validate
    print("\n" + "=" * 80)
    print("📊 Validation Summary")
    print("=" * 80)
    
    checks = []
    
    # Check 1: Embeddings are not all zeros
    is_not_zero = vision_embeddings.abs().sum() > 0
    checks.append(("Embeddings are non-zero", is_not_zero))
    
    # Check 2: Embeddings are finite
    is_finite = torch.isfinite(vision_embeddings).all()
    checks.append(("Embeddings are finite", is_finite))
    
    # Check 3: Expected shape (handle both 2D and 3D)
    if vision_embeddings.ndim == 3:
        batch_size, seq_len, hidden_dim = vision_embeddings.shape
    elif vision_embeddings.ndim == 2:
        seq_len, hidden_dim = vision_embeddings.shape
        batch_size = 1
        print(f"   Note: Vision embeddings are 2D, treating as batch_size=1")
    else:
        print(f"   ⚠️  Unexpected shape dimension: {vision_embeddings.ndim}")
        return False
    
    expected_hidden_dim = 2048  # Qwen3-VL Vision Encoder output dimension
    is_correct_dim = hidden_dim == expected_hidden_dim
    if not is_correct_dim:
        print(f"   ⚠️  Expected hidden_dim={expected_hidden_dim}, got {hidden_dim}")
    checks.append((f"Vision encoder hidden dimension is {expected_hidden_dim}", is_correct_dim))
    
    # Check 4: Vision positions found
    has_positions = len(vision_positions) > 0
    checks.append(("Vision token positions found", has_positions))
    
    for check_name, passed in checks:
        status = "✅" if passed else "❌"
        print(f"{status} {check_name}")
    
    all_passed = all(passed for _, passed in checks)
    
    print("\n" + "=" * 80)
    if all_passed:
        print("🎉 All checks passed! Vision embedding generation is working correctly.")
    else:
        print("⚠️  Some checks failed. Please review the results above.")
    print("=" * 80)
    
    return all_passed


def test_image_embedding(image_path: str, model_name: str = "Qwen/Qwen3-VL-2B-Instruct"):
    """
    Test vision embedding generation for an image input
    
    Args:
        image_path: Path to image file
        model_name: Model to use for processing
    """
    print("=" * 80)
    print("🧪 Testing Image Embedding Generation")
    print("=" * 80)
    
    # 1. Initialize components
    print("\n[Step 1/4] Initializing Preprocessor...")
    preprocessor = ClientPreprocessor(model_name)
    
    print("\n[Step 2/4] Initializing Vision Encoder...")
    vision_encoder = ClientVisionEncoder(model_name)
    
    # 2. Create test message with image
    print(f"\n[Step 3/4] Processing image: {image_path}")
    # Use absolute path without file:// prefix (as shown in cookbooks)
    import os
    image_abs_path = os.path.abspath(image_path)
    messages = [
        {
            "role": "user",
            "content": [
                {"image": image_abs_path},  # Direct path, no "type" or "file://" prefix
                {"type": "text", "text": "What do you see in this image?"}
            ]
        }
    ]
    
    # 3. Preprocess
    print("\n   📦 Preprocessing...")
    inputs = preprocessor.preprocess(messages)
    
    print("\n   ✅ Preprocessing Results:")
    print(f"      - input_ids shape: {inputs['input_ids'].shape}")
    print(f"      - attention_mask shape: {inputs['attention_mask'].shape}")
    
    if 'pixel_values' in inputs:
        print(f"      - pixel_values shape: {inputs['pixel_values'].shape}")
        print(f"      - pixel_values dtype: {inputs['pixel_values'].dtype}")
    
    if 'pixel_values_videos' in inputs:
        print(f"      - pixel_values_videos shape: {inputs['pixel_values_videos'].shape}")
        print(f"      - pixel_values_videos dtype: {inputs['pixel_values_videos'].dtype}")
    
    if 'image_grid_thw' in inputs:
        print(f"      - image_grid_thw shape: {inputs['image_grid_thw'].shape}")
        print(f"      - image_grid_thw values: {inputs['image_grid_thw']}")
    
    # 4. Encode vision
    print("\n[Step 4/4] Generating Vision Embeddings...")
    
    # Check for either pixel_values (images) or pixel_values_videos (videos)  
    pixel_values = inputs.get('pixel_values')
    if pixel_values is None:
        print("   ⚠️  No pixel_values found - image might not have been processed")
        return False
    
    vision_embeddings = vision_encoder.encode(
        pixel_values,
        inputs.get('image_grid_thw'),
        inputs.get('video_grid_thw')
    )
    
    print("\n   ✅ Vision Embedding Results:")
    print(f"      - Shape: {vision_embeddings.shape}")
    print(f"      - Dtype: {vision_embeddings.dtype}")
    print(f"      - Device: {vision_embeddings.device}")
    print(f"      - Min value: {vision_embeddings.min().item():.4f}")
    print(f"      - Max value: {vision_embeddings.max().item():.4f}")
    print(f"      - Mean value: {vision_embeddings.mean().item():.4f}")
    
    # 5. Get vision token positions
    vision_positions = preprocessor.extract_vision_token_positions(inputs['input_ids'])
    print(f"\n   🎯 Vision Token Positions: {vision_positions}")
    
    print("\n" + "=" * 80)
    print("✅ Image embedding generation complete!")
    print("=" * 80)
    
    return True


if __name__ == "__main__":
    # Usage examples
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python test_vision_embedding.py video <path_to_video>")
        print("  python test_vision_embedding.py image <path_to_image>")
        print("\nExample:")
        print("  python test_vision_embedding.py video ../cookbooks/assets/omni_recognition/video_example.mp4")
        print("  python test_vision_embedding.py image ../cookbooks/assets/omni_recognition/image_example.jpg")
        sys.exit(1)
    
    media_type = sys.argv[1]
    media_path = sys.argv[2]
    
    if media_type == "video":
        success = test_vision_embedding(media_path)
    elif media_type == "image":
        success = test_image_embedding(media_path)
    else:
        print(f"❌ Unknown media type: {media_type}")
        print("   Use 'video' or 'image'")
        sys.exit(1)
    
    sys.exit(0 if success else 1)
