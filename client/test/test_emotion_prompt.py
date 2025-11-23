"""
Test script to show the actual prompt with emotion injection
This demonstrates what the LLM actually sees
"""

import os
import sys
from simple_processor_wrapper import SimpleProcessorWrapper, format_emotion_for_prompt

# Test video
test_video = os.path.join(
    os.path.dirname(__file__),
    "../../examples/sad_woman.mp4"
)

print("=" * 80)
print("🧪 EMOTION-AWARE PROMPT TEST")
print("=" * 80)

# Initialize wrapper
wrapper = SimpleProcessorWrapper(device='cuda')

# Extract emotion
print("\n📹 Processing video:", test_video)
emotion_result = wrapper.extract_emotion_from_video(test_video)

print("\n" + "=" * 80)
print("📊 EMOTION ANALYSIS RESULT")
print("=" * 80)
print(f"Dominant Emotion: {emotion_result['dominant_emotion'].upper()}")
print(f"Sentiment Score: {emotion_result['sentiment']:.3f}")
print("\nAll Scores:")
for label, value in emotion_result['emotion_dict'].items():
    marker = "★" if label == emotion_result['dominant_emotion'] else " "
    print(f"  {marker} {label.capitalize()}: {value:.3f}")

# Format emotion for prompt
emotion_context = format_emotion_for_prompt(emotion_result)

# Example user message
user_message = "Hello, how are you?"

# Show comparison
print("\n" + "=" * 80)
print("📝 PROMPT COMPARISON")
print("=" * 80)

print("\n❌ WITHOUT EMOTION (normal prompt):")
print("-" * 80)
print(user_message)

print("\n✅ WITH EMOTION (emotion-aware prompt):")
print("-" * 80)
full_prompt = emotion_context + user_message
print(full_prompt)

print("\n" + "=" * 80)
print("📏 PROMPT STATISTICS")
print("=" * 80)
print(f"Original length: {len(user_message)} chars")
print(f"With emotion: {len(full_prompt)} chars")
print(f"Emotion context: {len(emotion_context)} chars")

print("\n" + "=" * 80)
print("💡 HOW TO VERIFY IN GRADIO")
print("=" * 80)
print("""
1. Start gradio_app.py and server
2. Upload sad_woman.mp4 as emotion video
3. Enable emotion recognition
4. Send a message
5. Check TERMINAL LOGS for:
   - "🎭 Injecting emotion context into prompt:"
   - "✅ Emotion context added to..."
   - Full prompt preview

6. Compare LLM responses:
   - Send message WITHOUT emotion video → Normal response
   - Send SAME message WITH emotion video → Should be more empathetic
""")

# Cleanup
wrapper.cleanup()

print("\n✅ Test complete!")
print("=" * 80)
