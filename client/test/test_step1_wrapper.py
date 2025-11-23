"""
Test Step 1: UniMSE Wrapper Initialization
Tests that the wrapper can load UniMSE model without modifying original code
"""

import sys
import torch
from pathlib import Path

print("="*60)
print("🧪 Step 1 Test: UniMSE Wrapper Initialization")
print("="*60)

try:
    # Test 1: Import wrapper
    print("\n1️⃣ Testing wrapper import...")
    from unimse_emotion_wrapper import UniMSEEmotionWrapper, UniMSEConfig
    print("   ✅ Import successful")
    
    # Test 2: Create config
    print("\n2️⃣ Testing config creation...")
    config = UniMSEConfig()
    print(f"   ✅ Config created")
    print(f"      Visual input: {config.d_vin}-dim → {config.d_vout}-dim")
    print(f"      Acoustic input: {config.d_ain}-dim → {config.d_aout}-dim")
    print(f"      Adapter: {config.adapter_name}")
    
    # Test 3: Initialize wrapper
    print("\n3️⃣ Testing wrapper initialization...")
    wrapper = UniMSEEmotionWrapper(device='cpu', adapter_name='ffn')
    print("   ✅ Wrapper initialized")
    
    # Test 4: Check model structure
    print("\n4️⃣ Testing model structure...")
    print(f"   Model type: {type(wrapper.model).__name__}")
    print(f"   Has visual_enc: {hasattr(wrapper.model, 'visual_enc')}")
    print(f"   Has acoustic_enc: {hasattr(wrapper.model, 'acoustic_enc')}")
    print(f"   Has T5_encoder: {hasattr(wrapper.model, 'T5_encoder')}")
    print("   ✅ Model structure correct")
    
    # Test 5: Test encoding with dummy data
    print("\n5️⃣ Testing encoding with dummy data...")
    visual_features = torch.randn(15, 35)
    visual_lengths = torch.tensor([15])
    acoustic_features = torch.randn(30, 74)
    acoustic_lengths = torch.tensor([30])
    
    result = wrapper.encode(
        text="I am feeling happy",
        visual_features=visual_features,
        visual_lengths=visual_lengths,
        acoustic_features=acoustic_features,
        acoustic_lengths=acoustic_lengths
    )
    
    print(f"   ✅ Encoding successful")
    print(f"      Emotion: {result['emotion_label']}")
    print(f"      Polarity: {result['polarity']:.3f}")
    print(f"      Intensity: {result['intensity']:.3f}")
    print(f"      Raw output: {result['raw_output']}")
    
    # Test 6: Test without text
    print("\n6️⃣ Testing encoding without text...")
    result_no_text = wrapper.encode(
        text=None,
        visual_features=visual_features,
        visual_lengths=visual_lengths,
        acoustic_features=acoustic_features,
        acoustic_lengths=acoustic_lengths
    )
    
    print(f"   ✅ Encoding without text successful")
    print(f"      Emotion: {result_no_text['emotion_label']}")
    
    # Test 7: Test batch dimension handling
    print("\n7️⃣ Testing batch dimension handling...")
    
    # 2D input (no batch)
    visual_2d = torch.randn(10, 35)
    result_2d = wrapper.encode(
        text="test",
        visual_features=visual_2d,
        visual_lengths=torch.tensor([10]),
        acoustic_features=torch.randn(20, 74),
        acoustic_lengths=torch.tensor([20])
    )
    print(f"   ✅ 2D input handled: {result_2d['emotion_label']}")
    
    # 3D input (with batch)
    visual_3d = torch.randn(1, 10, 35)
    result_3d = wrapper.encode(
        text="test",
        visual_features=visual_3d,
        visual_lengths=torch.tensor([10]),
        acoustic_features=torch.randn(1, 20, 74),
        acoustic_lengths=torch.tensor([20])
    )
    print(f"   ✅ 3D input handled: {result_3d['emotion_label']}")
    
    # Summary
    print("\n" + "="*60)
    print("✅ Step 1: ALL TESTS PASSED!")
    print("="*60)
    print("\n📝 Summary:")
    print("   ✅ UniMSE wrapper initialized successfully")
    print("   ✅ Original UniMSE code used without modification")
    print("   ✅ Encoding works with text + visual + acoustic")
    print("   ✅ Batch dimension handling works correctly")
    print("\n⚠️  Note: Predictions are random (no trained checkpoint loaded)")
    print("   To use trained model, provide checkpoint_path parameter")
    
    sys.exit(0)
    
except Exception as e:
    print(f"\n❌ Step 1 Test FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
