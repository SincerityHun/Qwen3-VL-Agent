"""
Test script for Simple Processor integration
Tests the wrapper independently before running full gradio app
"""

import os
import sys
import torch

# Test wrapper
print("="*60)
print("Testing Simple Processor Wrapper")
print("="*60)

from simple_processor_wrapper import SimpleProcessorWrapper, format_emotion_for_prompt, format_emotion_display

# Initialize wrapper
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\n🔧 Initializing wrapper with device: {device}")

wrapper = SimpleProcessorWrapper(device=device)

# Test with sample video
test_video = os.path.join(
    os.path.dirname(__file__),
    "../../examples/sad_woman.mp4"
)

if not os.path.exists(test_video):
    print(f"\n❌ Test video not found: {test_video}")
    print("Available videos in examples/:")
    examples_dir = os.path.join(os.path.dirname(__file__), "../../examples")
    if os.path.exists(examples_dir):
        for f in os.listdir(examples_dir):
            if f.endswith('.mp4'):
                print(f"  - {f}")
    sys.exit(1)

print(f"\n🎥 Testing with: {test_video}")
print("This may take a minute...")

try:
    result = wrapper.extract_emotion_from_video(test_video)
    
    print("\n" + "="*60)
    print("✅ Emotion Extraction Result")
    print("="*60)
    print(f"Dominant Emotion: {result['dominant_emotion']}")
    print(f"Sentiment Score: {result['sentiment']:.3f}")
    print("\nAll Emotion Scores:")
    for label, value in result['emotion_dict'].items():
        marker = "★" if label == result['dominant_emotion'] else " "
        print(f"  {marker} {label.capitalize()}: {value:.3f}")
    
    print("\n" + "="*60)
    print("Prompt Format (for LLM)")
    print("="*60)
    prompt_text = format_emotion_for_prompt(result)
    print(prompt_text)
    
    print("\n" + "="*60)
    print("Display Format (for UI)")
    print("="*60)
    display_text = format_emotion_display(result)
    print(display_text)
    
    print("\n" + "="*60)
    print("✅ Test Completed Successfully!")
    print("="*60)
    print("\nYou can now run: python gradio_app.py")
    
except Exception as e:
    print(f"\n❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
finally:
    wrapper.cleanup()
    print("\n🧹 Cleaned up temporary files")
