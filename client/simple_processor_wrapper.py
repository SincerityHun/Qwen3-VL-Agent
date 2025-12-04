"""
Simple Processor Wrapper
Wraps simple_processor Pipeline for easy integration with gradio_app
"""

import os
import sys
import torch
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Optional
import logging

# Add simple_processor to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'simple_processor'))

from simple_processor.pipeline import Pipeline

logger = logging.getLogger(__name__)


class SimpleProcessorWrapper:
    """
    Wrapper for simple_processor Pipeline
    Handles emotion extraction from video files
    """
    
    # Emotion labels matching the 7-dimensional output
    EMOTION_LABELS = ['sentiment', 'happy', 'sad', 'anger', 'surprise', 'disgust', 'fear']
    
    def __init__(self, device: str = 'cuda'):
        """
        Initialize Simple Processor
        
        Args:
            device: Device to run model on ('cuda' or 'cpu')
        """
        self.device = device
        self.temp_base_dir = None  # Base directory for all temp folders
        self.current_temp_dir = None  # Current video processing temp dir
        self.pipeline = None
        
        logger.info("🎭 Initializing Simple Processor for emotion recognition")
        
        # Create base temp directory
        self.temp_base_dir = tempfile.mkdtemp(prefix='simple_processor_base_')
        logger.info(f"📁 Created base temp directory: {self.temp_base_dir}")
        
        logger.info("✅ Simple Processor initialized")
    
    def _initialize_pipeline(self):
        """Initialize new pipeline with fresh temp directory for current video"""
        # Create new temporary directory for this specific video
        self.current_temp_dir = tempfile.mkdtemp(
            prefix='video_', 
            dir=self.temp_base_dir
        )
        logger.info(f"📁 Created new temp directory for video: {self.current_temp_dir}")
        
        # Change to simple_processor directory to ensure correct paths
        original_cwd = os.getcwd()
        simple_processor_dir = os.path.join(os.path.dirname(__file__), 'simple_processor')
        os.chdir(simple_processor_dir)
        
        try:
            # Initialize pipeline with fresh temp directory
            data_name = "emotion_data"
            self.pipeline = Pipeline(
                device=self.device,
                data_path=self.current_temp_dir,
                data_name=data_name
            )
            self.data_name = data_name
        finally:
            # Restore original working directory
            os.chdir(original_cwd)
    
    def extract_emotion_from_video(self, video_path: str) -> Dict:
        """
        Extract emotion from video file
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary containing:
                - emotion_values: List[float] of 7 emotion scores
                - emotion_labels: List[str] of emotion names
                - dominant_emotion: str of highest scoring emotion
                - emotion_dict: Dict mapping labels to values
        """
        try:
            logger.info(f"🎥 Processing video: {video_path}")
            
            # Create NEW pipeline for this video (ensures fresh data directory)
            self._initialize_pipeline()
            
            # Run pipeline
            with torch.no_grad():
                emotion_output = self.pipeline(video_path)
            
            # Check if pipeline returned None (processing failed)
            if emotion_output is None:
                logger.warning("⚠️  Pipeline returned None, using default neutral emotion")
                return self._get_default_emotion()
            
            # Convert to list
            if isinstance(emotion_output, torch.Tensor):
                emotion_values = emotion_output.cpu().numpy().tolist()
            else:
                emotion_values = list(emotion_output)
            
            # DEBUG: Log raw emotion output
            logger.info(f"🔍 Raw emotion output: {emotion_values}")
            
            # Ensure we have exactly 7 values
            if len(emotion_values) != 7:
                logger.warning(f"⚠️  Expected 7 emotion values, got {len(emotion_values)}")
                emotion_values = emotion_values[:7] + [0.0] * (7 - len(emotion_values))
            
            # Create emotion dictionary
            emotion_dict = {
                label: float(value) 
                for label, value in zip(self.EMOTION_LABELS, emotion_values)
            }
            
            # Find dominant emotion (excluding sentiment)
            emotion_scores = emotion_values[1:]  # Skip sentiment
            emotion_names = self.EMOTION_LABELS[1:]
            dominant_idx = emotion_scores.index(max(emotion_scores))
            dominant_emotion = emotion_names[dominant_idx]
            
            # Calculate top-3 emotions with scores
            emotion_pairs = list(zip(emotion_names, emotion_scores))
            sorted_emotions = sorted(emotion_pairs, key=lambda x: x[1], reverse=True)
            top_3_emotions = sorted_emotions[:3]
            
            # Calculate confidence (difference between top 1 and top 2)
            max_score = sorted_emotions[0][1]
            second_score = sorted_emotions[1][1]
            confidence = abs(max_score - second_score)
            
            # Determine if emotions are mixed (top 3 scores are close)
            score_range = max_score - sorted_emotions[2][1]
            is_mixed = score_range < 0.05  # If range < 0.05, consider mixed
            
            result = {
                'emotion_values': emotion_values,
                'emotion_labels': self.EMOTION_LABELS,
                'dominant_emotion': dominant_emotion,
                'emotion_dict': emotion_dict,
                'sentiment': emotion_values[0],
                'top_3_emotions': top_3_emotions,  # [(name, score), ...]
                'confidence': confidence,  # How confident is dominant emotion
                'is_mixed': is_mixed,  # Are emotions mixed?
                'all_emotions_ranked': sorted_emotions  # Full ranking
            }
            
            logger.info(f"✅ Emotion extracted: {dominant_emotion} (conf={confidence:.4f}, mixed={is_mixed})")
            logger.info(f"   Top 3: {', '.join([f'{name}({score:.4f})' for name, score in top_3_emotions])}")
            
            return result
            
        except Exception as e:
            logger.warning(f"⚠️  Emotion extraction failed: {e}")
            logger.info("Using default neutral emotion values")
            return self._get_default_emotion()
    
    def _get_default_emotion(self) -> Dict:
        """
        Return default neutral emotion values when extraction fails
        
        Returns:
            Dictionary with neutral emotion values
        """
        # EMOTION_LABELS = ['sentiment', 'happy', 'sad', 'anger', 'surprise', 'disgust', 'fear']
        # Index:              0          1        2      3        4           5          6
        # Default neutral values: equal distribution across all emotions
        default_values = [0.0, 0.17, 0.17, 0.17, 0.17, 0.17, 0.15]  # Sums to ~1.0 for emotions
        
        emotion_dict = {
            label: float(value) 
            for label, value in zip(self.EMOTION_LABELS, default_values)
        }
        
        # Extract actual emotions (skip sentiment at index 0)
        emotion_scores = default_values[1:]  # [0.17, 0.17, 0.17, 0.17, 0.17, 0.15]
        emotion_names = self.EMOTION_LABELS[1:]  # ['happy', 'sad', 'anger', 'surprise', 'disgust', 'fear']
        
        # Create emotion pairs and sort
        emotion_pairs = list(zip(emotion_names, emotion_scores))
        sorted_emotions = sorted(emotion_pairs, key=lambda x: x[1], reverse=True)
        top_3_emotions = sorted_emotions[:3]
        
        # All emotions equal, so confidence is 0
        confidence = 0.0
        is_mixed = True  # Default is always mixed (all equal)
        
        result = {
            'emotion_values': default_values,
            'emotion_labels': self.EMOTION_LABELS,
            'dominant_emotion': 'happy',  # First emotion after sentiment
            'emotion_dict': emotion_dict,
            'sentiment': 0.0,  # default_values[0]
            'top_3_emotions': top_3_emotions,
            'confidence': confidence,
            'is_mixed': is_mixed,
            'all_emotions_ranked': sorted_emotions,
            'is_default': True  # Flag to indicate this is a fallback
        }
        
        logger.info("ℹ️  Using default neutral emotion values")
        return result
    
    def cleanup(self):
        """Cleanup all temporary directories"""
        if self.temp_base_dir and os.path.exists(self.temp_base_dir):
            try:
                shutil.rmtree(self.temp_base_dir)
                logger.info(f"🧹 Cleaned up base temp directory: {self.temp_base_dir}")
            except Exception as e:
                logger.warning(f"⚠️  Could not cleanup temp directory: {e}")
            finally:
                self.temp_base_dir = None
                self.current_temp_dir = None
    
    def __del__(self):
        """Cleanup on deletion"""
        self.cleanup()


def format_emotion_for_prompt(emotion_result: Dict) -> str:
    """
    Format emotion result for LLM prompt with full emotion distribution
    
    Args:
        emotion_result: Result from extract_emotion_from_video
        
    Returns:
        Formatted string for prompt
    """
    emotion_dict = emotion_result['emotion_dict']
    dominant = emotion_result['dominant_emotion']
    sentiment = emotion_result['sentiment']
    top_3 = emotion_result.get('top_3_emotions', [])
    confidence = emotion_result.get('confidence', 0.0)
    is_mixed = emotion_result.get('is_mixed', False)
    
    # Determine sentiment category
    sentiment_desc = "positive" if sentiment > 0.1 else "negative" if sentiment < -0.1 else "neutral"
    
    # Build emotion state description
    if is_mixed and len(top_3) >= 3:
        # Mixed emotions - describe top 3
        emotion_state = f"{top_3[0][0]} ({top_3[0][1]:.3f}), {top_3[1][0]} ({top_3[1][1]:.3f}), and {top_3[2][0]} ({top_3[2][1]:.3f})"
        emotion_desc = f"The user is experiencing MIXED emotions: primarily {emotion_state}. The emotions are closely balanced (confidence: {confidence:.3f})."
    else:
        # Clear dominant emotion
        if len(top_3) >= 2:
            secondary = f", with some {top_3[1][0]} ({top_3[1][1]:.3f})"
        else:
            secondary = ""
        emotion_desc = f"The user is clearly feeling {dominant} ({top_3[0][1]:.3f} score){secondary}. Confidence: {confidence:.3f}."
    
    # Emotion-specific response guidelines
    emotion_guidelines = {
        'happy': "Use an upbeat, cheerful tone. Engage positively and celebrate with them.",
        'sad': "Use a warm, supportive tone. Acknowledge their feelings gently and offer comfort.",
        'anger': "Use a calm, understanding tone. Validate their frustration and offer constructive solutions.",
        'fear': "Use a reassuring, clear tone. Provide step-by-step guidance and emphasize safety.",
        'surprise': "Use an informative, patient tone. Provide clear explanations.",
        'disgust': "Use a respectful, constructive tone. Acknowledge concerns and suggest alternatives.",
    }
    
    # Get guidelines for all relevant emotions
    if is_mixed and len(top_3) >= 2:
        guidelines = []
        for emotion_name, score in top_3[:2]:
            guideline = emotion_guidelines.get(emotion_name, "")
            if guideline:
                guidelines.append(f"- For {emotion_name} aspect: {guideline}")
        guideline_text = "\n".join(guidelines)
    else:
        guideline_text = emotion_guidelines.get(dominant, "Use a balanced, helpful tone.")
    
    # Create comprehensive emotion-aware prompt
    prompt_text = f"""<emotional_context>
User's Emotional State Analysis:
{emotion_desc}
Sentiment: {sentiment_desc} ({sentiment:+.3f})

Full Emotion Distribution:
{chr(10).join([f'- {name}: {score:.4f}' for name, score in top_3])}

Response Guidelines:
{guideline_text}

IMPORTANT:
1. Answer their question directly and completely
2. Adapt your tone based on the emotional context above
3. If emotions are mixed, acknowledge the complexity of their feelings
4. Be empathetic and emotionally intelligent in your response
</emotional_context>

User's question: """
    
    return prompt_text


def format_emotion_display(emotion_result: Dict) -> str:
    """
    Format emotion for display in Gradio UI
    
    Args:
        emotion_result: Result from extract_emotion_from_video
        
    Returns:
        Markdown formatted string for display
    """
    emotion_dict = emotion_result['emotion_dict']
    dominant = emotion_result['dominant_emotion']
    sentiment = emotion_result['sentiment']
    
    # Emoji mapping
    emoji_map = {
        'happy': '😊',
        'sad': '😢',
        'anger': '😠',
        'fear': '😰',
        'surprise': '😲',
        'disgust': '🤢',
    }
    
    emoji = emoji_map.get(dominant, '🙂')
    sentiment_emoji = '😊' if sentiment > 0 else '😢' if sentiment < 0 else '😐'
    
    # Get additional info
    top_3 = emotion_result.get('top_3_emotions', [])
    confidence = emotion_result.get('confidence', 0.0)
    is_mixed = emotion_result.get('is_mixed', False)
    
    # Create bar chart visualization with ranking
    bars = []
    all_emotions_ranked = emotion_result.get('all_emotions_ranked', [])
    if all_emotions_ranked:
        for idx, (label, value) in enumerate(all_emotions_ranked, 1):
            bar_length = int(abs(value) * 30)  # Scale to 30 chars max
            bar = '█' * bar_length
            marker = '🥇' if idx == 1 else '🥈' if idx == 2 else '🥉' if idx == 3 else '  '
            bars.append(f"{marker} **{label.capitalize()}**: {bar} {value:.4f}")
    else:
        # Fallback to original
        for label in ['happy', 'sad', 'anger', 'surprise', 'disgust', 'fear']:
            value = emotion_dict[label]
            bar_length = int(abs(value) * 30)
            bar = '█' * bar_length
            marker = '★' if label == dominant else '  '
            bars.append(f"{marker} **{label.capitalize()}**: {bar} {value:.4f}")
    
    bars_text = "\n".join(bars)
    
    # Emotion state description
    if is_mixed:
        state_desc = f"**MIXED EMOTIONS** (Confidence: {confidence:.4f})\n"
        state_desc += f"Top emotions: {', '.join([f'{name} ({score:.3f})' for name, score in top_3[:3]])}"
    else:
        state_desc = f"**Clear {dominant.upper()}** (Confidence: {confidence:.4f})"
    
    display = f"""## 🎭 Emotion Analysis

### {emoji} {state_desc}

**Sentiment**: {sentiment_emoji} {sentiment:+.3f} ({'Positive' if sentiment > 0 else 'Negative' if sentiment < 0 else 'Neutral'})

### Emotion Distribution (Ranked):
{bars_text}

---
💡 *The AI responds with full awareness of your emotional state, including mixed feelings*
"""
    
    return display


if __name__ == "__main__":
    # Test wrapper
    print("🧪 Testing SimpleProcessorWrapper\n")
    
    wrapper = SimpleProcessorWrapper(device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test with sample video
    test_video = os.path.join(
        os.path.dirname(__file__),
        "../../examples/IronMan.mp4"
    )
    
    if os.path.exists(test_video):
        print(f"Testing with: {test_video}")
        result = wrapper.extract_emotion_from_video(test_video)
        
        print("\n" + "="*60)
        print("Emotion Result:")
        print("="*60)
        print(f"Dominant: {result['dominant_emotion']}")
        print(f"Sentiment: {result['sentiment']:.3f}")
        print("\nAll scores:")
        for label, value in result['emotion_dict'].items():
            print(f"  {label}: {value:.3f}")
        
        print("\n" + "="*60)
        print("Prompt Format:")
        print("="*60)
        print(format_emotion_for_prompt(result))
        
        print("\n" + "="*60)
        print("Display Format:")
        print("="*60)
        print(format_emotion_display(result))
        
    else:
        print(f"❌ Test video not found: {test_video}")
    
    wrapper.cleanup()
    print("\n✅ Test completed!")
