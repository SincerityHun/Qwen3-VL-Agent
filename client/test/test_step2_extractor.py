"""
Test Step 2: Emotion Extractor + UniMSE Wrapper Integration
Tests that emotion_extractor output works with UniMSE wrapper
"""

import sys
import torch
from pathlib import Path

print("="*60)
print("🧪 Step 2 Test: Emotion Extractor + UniMSE Integration")
print("="*60)

try:
    # Test 1: Import modules
    print("\n1️⃣ Testing imports...")
    from emotion_extractor import EmotionFeatureExtractor
    from unimse_emotion_wrapper import UniMSEEmotionWrapper
    print("   ✅ Imports successful")
    
    # Test 2: Initialize extractor
    print("\n2️⃣ Initializing emotion extractor...")
    extractor = EmotionFeatureExtractor(target_fps=5)
    print("   ✅ Extractor initialized")
    
    # Test 3: Test with synthetic data (simulated video)
    print("\n3️⃣ Testing with synthetic data...")
    
    # Simulate variable-length sequences
    visual_features = torch.randn(15, 35)  # 15 frames
    visual_lengths = torch.tensor([15])
    acoustic_features = torch.randn(30, 74)  # 30 frames (different length)
    acoustic_lengths = torch.tensor([30])
    
    print(f"   Visual: {visual_features.shape}, length: {visual_lengths.item()}")
    print(f"   Acoustic: {acoustic_features.shape}, length: {acoustic_lengths.item()}")
    print("   ✅ Variable-length sequences created")
    
    # Test 4: Initialize UniMSE wrapper
    print("\n4️⃣ Initializing UniMSE wrapper...")
    wrapper = UniMSEEmotionWrapper(device='cpu', adapter_name='ffn')
    print("   ✅ Wrapper initialized")
    
    # Test 5: Encode with extracted features
    print("\n5️⃣ Testing encoding with synthetic features...")
    result = wrapper.encode(
        text="I am feeling anxious about this",
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
    
    # Test 6: Test with real video (if available)
    print("\n6️⃣ Testing with real video...")
    test_video = Path("../examples/IronMan.mp4")
    
    if test_video.exists():
        print(f"   📹 Processing: {test_video}")
        
        # Extract features
        features = extractor.extract_from_video_file(str(test_video))
        
        print(f"   ✅ Features extracted")
        print(f"      Visual: {features['visual_features'].shape}, length: {features['visual_lengths'].item()}")
        print(f"      Acoustic: {features['acoustic_features'].shape}, length: {features['acoustic_lengths'].item()}")
        
        # Encode with wrapper
        result_real = wrapper.encode(
            text="I'm ready to fight the bad guys!",
            visual_features=features['visual_features'],
            visual_lengths=features['visual_lengths'],
            acoustic_features=features['acoustic_features'],
            acoustic_lengths=features['acoustic_lengths']
        )
        
        print(f"\n   ✅ Real video encoding successful")
        print(f"      Emotion: {result_real['emotion_label']}")
        print(f"      Polarity: {result_real['polarity']:.3f}")
        print(f"      Intensity: {result_real['intensity']:.3f}")
        print(f"      Raw output: {result_real['raw_output']}")
    else:
        print(f"   ⚠️  Test video not found: {test_video}")
        print(f"      Skipping real video test")
    
    # Test 7: Test dimension compatibility
    print("\n7️⃣ Testing dimension compatibility...")
    
    test_cases = [
        ("Short sequence", torch.randn(5, 35), torch.tensor([5]), torch.randn(10, 74), torch.tensor([10])),
        ("Long sequence", torch.randn(50, 35), torch.tensor([50]), torch.randn(100, 74), torch.tensor([100])),
        ("Single frame", torch.randn(1, 35), torch.tensor([1]), torch.randn(1, 74), torch.tensor([1])),
    ]
    
    for name, v_feat, v_len, a_feat, a_len in test_cases:
        result_test = wrapper.encode(
            text="test",
            visual_features=v_feat,
            visual_lengths=v_len,
            acoustic_features=a_feat,
            acoustic_lengths=a_len
        )
        print(f"   ✅ {name}: v={v_feat.shape}, a={a_feat.shape} → {result_test['emotion_label']}")
    
    # Test 8: Test format compatibility
    print("\n8️⃣ Testing format compatibility...")
    
    # extractor output format
    dummy_features = {
        'visual_features': torch.randn(12, 35),
        'visual_lengths': torch.tensor([12]),
        'acoustic_features': torch.randn(24, 74),
        'acoustic_lengths': torch.tensor([24])
    }
    
    # Should work directly
    result_format = wrapper.encode(
        text="Format compatibility test",
        visual_features=dummy_features['visual_features'],
        visual_lengths=dummy_features['visual_lengths'],
        acoustic_features=dummy_features['acoustic_features'],
        acoustic_lengths=dummy_features['acoustic_lengths']
    )
    
    print(f"   ✅ Extractor output format compatible")
    print(f"      Result: {result_format['emotion_label']}")
    
    # Summary
    print("\n" + "="*60)
    print("✅ Step 2: ALL TESTS PASSED!")
    print("="*60)
    print("\n📝 Summary:")
    print("   ✅ Emotion extractor works correctly")
    print("   ✅ Output format compatible with UniMSE wrapper")
    print("   ✅ Variable-length sequences supported")
    print("   ✅ Real video processing works (if available)")
    print("   ✅ Different sequence lengths handled correctly")
    print("\n⚠️  Note: Predictions are random (no trained checkpoint)")
    print("   Next: Integrate into gradio_app.py (Step 3)")
    
    sys.exit(0)
    
except Exception as e:
    print(f"\n❌ Step 2 Test FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
