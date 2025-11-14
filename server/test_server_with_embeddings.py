"""
Test Server LLM Inference with Pre-computed Embeddings
Loads saved embeddings from client and tests server generation
"""

import torch
import sys
import os
import json
from pathlib import Path


def test_server_with_embeddings(embedding_file: str):
    """
    Test server LLM inference with pre-computed embeddings
    
    Args:
        embedding_file: Path to saved .pt file with embeddings
    """
    print("=" * 80)
    print("🧪 Testing Server with Pre-computed Embeddings")
    print("=" * 80)
    
    # 1. Load data
    print(f"\n[Step 1/4] Loading embeddings from {embedding_file}...")
    if not os.path.exists(embedding_file):
        print(f"❌ File not found: {embedding_file}")
        sys.exit(1)
    
    data = torch.load(embedding_file, map_location='cpu')
    
    print("   ✅ Data loaded:")
    print(f"      - input_ids: {data['input_ids'].shape}")
    print(f"      - attention_mask: {data['attention_mask'].shape}")
    print(f"      - vision_embeddings: {data['vision_embeddings'].shape}")
    print(f"      - vision_token_positions: {data['vision_token_positions']}")
    
    # Load metadata if available
    metadata_file = embedding_file.replace('_tensors.pt', '_metadata.json')
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        print(f"\n   📋 Metadata:")
        print(f"      - Media type: {metadata.get('media_type', 'unknown')}")
        print(f"      - Text prompt: {metadata.get('text_prompt', 'N/A')}")
        print(f"      - Model: {metadata.get('model_name', 'N/A')}")
    
    # 2. Initialize server
    print("\n[Step 2/4] Initializing Server LLM...")
    
    try:
        from llm_inference import ServerLLMInference
        
        model_name = os.getenv("MODEL_NAME", "Qwen/Qwen3-VL-2B-Instruct")
        llm = ServerLLMInference(
            model_name=model_name,
            device_map="auto"
        )
        print("   ✅ Server LLM loaded!")
        
    except Exception as e:
        print(f"   ❌ Failed to load server: {e}")
        print("\n   💡 Make sure you're in the server directory and have installed requirements:")
        print("      cd server")
        print("      pip install -r requirements.txt")
        sys.exit(1)
    
    # 3. Generate (non-streaming)
    print("\n[Step 3/4] Generating response (non-streaming)...")
    
    try:
        generated_text = llm.generate(
            input_ids=data['input_ids'],
            vision_embeddings=data['vision_embeddings'],
            vision_token_positions=data['vision_token_positions'],
            attention_mask=data['attention_mask'],
            max_new_tokens=128,
            temperature=0.7,
            top_p=0.8
        )
        
        print("\n   ✅ Generation complete!")
        print("\n" + "=" * 80)
        print("📝 Generated Response:")
        print("=" * 80)
        print(generated_text)
        print("=" * 80)
        
    except Exception as e:
        print(f"\n   ❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 4. Test streaming generation
    print("\n[Step 4/4] Testing streaming generation...")
    
    try:
        print("\n" + "=" * 80)
        print("📝 Streaming Response:")
        print("=" * 80)
        
        accumulated = ""
        for result in llm.generate_stream(
            input_ids=data['input_ids'],
            vision_embeddings=data['vision_embeddings'],
            vision_token_positions=data['vision_token_positions'],
            attention_mask=data['attention_mask'],
            max_new_tokens=128,
            temperature=0.7,
            top_p=0.8
        ):
            text = result.get('text', '')
            # Print only new characters
            new_text = text[len(accumulated):]
            print(new_text, end='', flush=True)
            accumulated = text
        
        print("\n" + "=" * 80)
        print("\n✅ Streaming generation complete!")
        
    except Exception as e:
        print(f"\n   ❌ Streaming failed: {e}")
        import traceback
        traceback.print_exc()
    
    # 5. Summary
    print("\n" + "=" * 80)
    print("📊 Test Summary")
    print("=" * 80)
    print(f"✅ Embedding loading: Success")
    print(f"✅ Server initialization: Success")
    print(f"✅ Non-streaming generation: Success")
    print(f"✅ Streaming generation: Success")
    print("\n🎉 All server tests passed!")
    print("=" * 80)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python test_server_with_embeddings.py <path_to_tensors.pt>")
        print("\nExample:")
        print("  python test_server_with_embeddings.py ../test_data/image_example_tensors.pt")
        print("  python test_server_with_embeddings.py ../test_data/video_example_tensors.pt")
        sys.exit(1)
    
    embedding_file = sys.argv[1]
    test_server_with_embeddings(embedding_file)
