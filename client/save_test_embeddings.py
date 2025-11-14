"""
Save test embeddings for server testing
Generates and saves image/video embeddings with all necessary metadata
"""

import torch
import json
import os
from preprocessor import ClientPreprocessor
from vision_encoder import ClientVisionEncoder
from pathlib import Path


def save_embedding_data(
    media_path: str,
    media_type: str,
    output_dir: str = "../test_data",
    model_name: str = "Qwen/Qwen3-VL-2B-Instruct"
):
    """
    Process media and save all data needed for server testing
    
    Args:
        media_path: Path to image or video file
        media_type: 'image' or 'video'
        output_dir: Directory to save test data
        model_name: Model to use
    """
    print("=" * 80)
    print(f"💾 Saving {media_type.upper()} Embedding Data for Server Testing")
    print("=" * 80)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Initialize components
    print("\n[Step 1/5] Initializing preprocessor and encoder...")
    preprocessor = ClientPreprocessor(model_name)
    vision_encoder = ClientVisionEncoder(model_name)
    
    # 2. Create messages
    print(f"\n[Step 2/5] Creating messages for {media_type}...")
    media_abs_path = os.path.abspath(media_path)
    
    if media_type == "image":
        messages = [
            {
                "role": "user",
                "content": [
                    {"image": media_abs_path},
                    {"type": "text", "text": "이 이미지를 자세히 설명해주세요"}
                ]
            }
        ]
    elif media_type == "video":
        messages = [
            {
                "role": "user",
                "content": [
                    {"video": media_abs_path},
                    {"type": "text", "text": "이 영상의 내용을 바탕으로 현재 감정을 분석해주세요"}
                ]
            }
        ]
    else:
        raise ValueError(f"Unknown media type: {media_type}")
    
    # 3. Preprocess
    print("\n[Step 3/5] Preprocessing...")
    inputs = preprocessor.preprocess(messages)
    
    print(f"   ✅ Preprocessing complete:")
    print(f"      - input_ids: {inputs['input_ids'].shape}")
    print(f"      - attention_mask: {inputs['attention_mask'].shape}")
    
    # 4. Encode vision
    print("\n[Step 4/5] Encoding vision features...")
    
    # Select correct pixel values
    if 'pixel_values_videos' in inputs:
        pixel_values = inputs['pixel_values_videos']
        print(f"      - Using pixel_values_videos: {pixel_values.shape}")
    elif 'pixel_values' in inputs:
        pixel_values = inputs['pixel_values']
        print(f"      - Using pixel_values: {pixel_values.shape}")
    else:
        raise ValueError("No pixel values found")
    
    vision_embeddings = vision_encoder.encode(
        pixel_values,
        inputs.get('image_grid_thw'),
        inputs.get('video_grid_thw')
    )
    
    print(f"   ✅ Vision embeddings: {vision_embeddings.shape}")
    
    # 5. Get vision token positions
    vision_positions = preprocessor.extract_vision_token_positions(inputs['input_ids'])
    print(f"      - Vision token positions: {vision_positions}")
    
    # 6. Save all data
    print("\n[Step 5/5] Saving data...")
    
    # Determine output filename
    base_name = Path(media_path).stem
    output_prefix = os.path.join(output_dir, f"{media_type}_{base_name}")
    
    # Save tensors
    torch.save({
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs['attention_mask'],
        'vision_embeddings': vision_embeddings,
        'vision_token_positions': vision_positions,
    }, f"{output_prefix}_tensors.pt")
    print(f"   ✅ Saved: {output_prefix}_tensors.pt")
    
    # Save metadata as JSON
    metadata = {
        'media_type': media_type,
        'media_path': media_path,
        'model_name': model_name,
        'input_ids_shape': list(inputs['input_ids'].shape),
        'attention_mask_shape': list(inputs['attention_mask'].shape),
        'vision_embeddings_shape': list(vision_embeddings.shape),
        'vision_token_positions': vision_positions,
        'text_prompt': messages[0]['content'][-1]['text'],
        'hidden_dim': vision_embeddings.shape[-1],
        'num_vision_tokens': len(vision_positions),
    }
    
    if 'image_grid_thw' in inputs:
        metadata['image_grid_thw'] = inputs['image_grid_thw'].tolist()
    if 'video_grid_thw' in inputs:
        metadata['video_grid_thw'] = inputs['video_grid_thw'].tolist()
    
    with open(f"{output_prefix}_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"   ✅ Saved: {output_prefix}_metadata.json")
    
    # Save human-readable info
    info_text = f"""
Test Data Summary
================================================================================
Media Type: {media_type}
Media Path: {media_path}
Model: {model_name}

Tensor Shapes:
- input_ids: {inputs['input_ids'].shape}
- attention_mask: {inputs['attention_mask'].shape}
- vision_embeddings: {vision_embeddings.shape}

Vision Token Positions: {vision_positions}
Number of Vision Tokens: {len(vision_positions)}

Text Prompt:
"{messages[0]['content'][-1]['text']}"

Statistics:
- Vision embedding mean: {vision_embeddings.mean().item():.6f}
- Vision embedding std: {vision_embeddings.std().item():.6f}
- Vision embedding min: {vision_embeddings.min().item():.6f}
- Vision embedding max: {vision_embeddings.max().item():.6f}

Files Saved:
- {output_prefix}_tensors.pt (PyTorch tensors)
- {output_prefix}_metadata.json (metadata)
- {output_prefix}_info.txt (this file)

Usage in Server Test:
--------------------
data = torch.load('{output_prefix}_tensors.pt')
input_ids = data['input_ids']
vision_embeddings = data['vision_embeddings']
vision_positions = data['vision_token_positions']
attention_mask = data['attention_mask']
================================================================================
"""
    
    with open(f"{output_prefix}_info.txt", 'w') as f:
        f.write(info_text)
    print(f"   ✅ Saved: {output_prefix}_info.txt")
    
    print("\n" + "=" * 80)
    print("🎉 All data saved successfully!")
    print("=" * 80)
    print(f"\nSaved files:")
    print(f"  - {output_prefix}_tensors.pt")
    print(f"  - {output_prefix}_metadata.json")
    print(f"  - {output_prefix}_info.txt")
    print(f"\nTest this data with:")
    print(f"  cd ../server")
    print(f"  python test_server_with_embeddings.py {output_prefix}_tensors.pt")
    
    return output_prefix


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python save_test_embeddings.py <media_type> <media_path> [output_dir]")
        print("\nExamples:")
        print("  python save_test_embeddings.py image ../cookbooks/assets/omni_recognition/image_example.jpg")
        print("  python save_test_embeddings.py video ../cookbooks/assets/omni_recognition/video_example.mp4")
        print("  python save_test_embeddings.py image test.jpg ../test_data")
        sys.exit(1)
    
    media_type = sys.argv[1]
    media_path = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "../pt_data_examples"
    
    if media_type not in ['image', 'video']:
        print(f"❌ Invalid media type: {media_type}")
        print("   Use 'image' or 'video'")
        sys.exit(1)
    
    if not os.path.exists(media_path):
        print(f"❌ Media file not found: {media_path}")
        sys.exit(1)
    
    save_embedding_data(media_path, media_type, output_dir)
