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
            
            result = {
                'emotion_values': emotion_values,
                'emotion_labels': self.EMOTION_LABELS,
                'dominant_emotion': dominant_emotion,
                'emotion_dict': emotion_dict,
                'sentiment': emotion_values[0]
            }
            
            logger.info(f"✅ Emotion extracted: {dominant_emotion} (sentiment={emotion_values[0]:.3f})")
            
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
        # Default neutral values: [sentiment=0, happy=0.14, sad=0.14, anger=0.14, surprise=0.14, disgust=0.14, fear=0.14]
        # Equal distribution across all emotions except sentiment
        default_values = [0.0, 0.17, 0.17, 0.17, 0.17, 0.17, 0.15]  # Sums to ~1.0 for emotions
        
        emotion_dict = {
            label: float(value) 
            for label, value in zip(self.EMOTION_LABELS, default_values)
        }
        
        result = {
            'emotion_values': default_values,
            'emotion_labels': self.EMOTION_LABELS,
            'dominant_emotion': 'happy',  # Default to happy (first emotion)
            'emotion_dict': emotion_dict,
            'sentiment': 0.0,
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
    Format emotion result for LLM prompt
    
    Args:
        emotion_result: Result from extract_emotion_from_video
        
    Returns:
        Formatted string for prompt
    """
    emotion_dict = emotion_result['emotion_dict']
    dominant = emotion_result['dominant_emotion']
    sentiment = emotion_result['sentiment']
    
    # Determine sentiment category
    sentiment_desc = "positive" if sentiment > 0.1 else "negative" if sentiment < -0.1 else "neutral"
    
    # Emotion-specific response guidelines
    emotion_guidelines = {
        'happy': "Use an upbeat, cheerful tone. Include positive suggestions like outdoor activities, celebrations, or fun ideas. You can add jokes or playful comments.",
        'sad': "Use a warm, supportive tone. Acknowledge their feelings gently. Suggest comforting activities like short walks, self-care, or calming routines. Be encouraging but not overly cheerful.",
        'anger': "Use a calm, understanding tone. Avoid being defensive. Focus on constructive solutions and validate their frustration. Offer practical help.",
        'fear': "Use a reassuring, clear tone. Provide step-by-step guidance. Emphasize safety and support. Help them feel more in control.",
        'surprise': "Use an informative, patient tone. Provide clear explanations. Acknowledge their curiosity. Break down complex information.",
        'disgust': "Use a respectful, constructive tone. Acknowledge their concerns. Suggest alternatives or solutions. Focus on improvement.",
    }
    
    guideline = emotion_guidelines.get(dominant, "Use a balanced, helpful tone.")
    
    # Create instruction-style prompt that doesn't override the user's question
    prompt_text = f"""[IMPORTANT CONTEXT: The user's video shows they are currently feeling {dominant} (sentiment: {sentiment_desc}). 

Response Guidelines:
1. Answer their question/request directly and completely
2. Adapt your tone and style: {guideline}
3. After answering, you may add brief emotional support or suggestions appropriate to their {dominant} state
4. Keep the focus on helping with their actual question while being emotionally aware]

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
    
    # Create bar chart visualization
    bars = []
    for label in ['happy', 'sad', 'anger', 'surprise', 'disgust', 'fear']:
        value = emotion_dict[label]
        bar_length = int(value * 20)  # Scale to 20 chars max
        bar = '█' * bar_length
        marker = '★' if label == dominant else ' '
        bars.append(f"{marker} **{label.capitalize()}**: {bar} {value:.3f}")
    
    bars_text = "\n".join(bars)
    
    display = f"""## 🎭 Emotion Analysis

### {emoji} Dominant Emotion: **{dominant.upper()}**

**Sentiment**: {sentiment_emoji} {sentiment:+.3f} ({'Positive' if sentiment > 0 else 'Negative' if sentiment < 0 else 'Neutral'})

### Emotion Breakdown:
{bars_text}

---
💡 *The AI will respond with awareness of your emotional state*
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
