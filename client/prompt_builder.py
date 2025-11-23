"""
Emotion-aware Prompt Builder
Injects emotion state into LLM prompts for empathetic responses
"""

from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmotionAwarePromptBuilder:
    """
    Build emotion-aware prompts for LLM conditioning
    
    Injects user's emotional context into the prompt to guide
    the LLM toward more empathetic and appropriate responses.
    """
    
    # Emotion-specific response guidelines
    EMOTION_GUIDELINES = {
        'joy': "The user seems happy and excited. Respond with enthusiasm and positivity.",
        'sadness': "The user appears sad or down. Respond with empathy, comfort, and supportive language.",
        'anger': "The user seems frustrated or angry. Respond calmly, avoid defensive language, and offer helpful solutions.",
        'fear': "The user appears anxious or worried. Respond with reassurance and clear, step-by-step guidance.",
        'surprise': "The user seems surprised or curious. Respond with detailed explanations and context.",
        'disgust': "The user appears displeased. Acknowledge their concerns and provide constructive alternatives.",
        'neutral': "The user has a neutral emotional state. Respond with balanced, informative tone."
    }
    
    POLARITY_GUIDELINES = {
        'very_positive': "very positive and enthusiastic",
        'positive': "positive and supportive",
        'neutral': "balanced and objective",
        'negative': "patient and understanding",
        'very_negative': "extremely empathetic and calming"
    }
    
    def __init__(self, include_embedding: bool = False):
        """
        Initialize prompt builder
        
        Args:
            include_embedding: Whether to include emotion embedding in context
        """
        self.include_embedding = include_embedding
        logger.info("✅ EmotionAwarePromptBuilder initialized")
    
    def build_emotion_context(self, emotion_state: Dict[str, any]) -> str:
        """
        Build emotion context string
        
        Args:
            emotion_state: Dictionary with keys:
                - polarity: float [-1, 1]
                - intensity: float [0, 1]
                - emotion_label: str
                - emotion_embedding: List[float] (optional)
        
        Returns:
            Formatted emotion context string
        """
        polarity = emotion_state['polarity']
        intensity = emotion_state['intensity']
        emotion = emotion_state['emotion_label']
        
        # Classify polarity level
        if polarity > 0.5:
            polarity_level = 'very_positive'
        elif polarity > 0:
            polarity_level = 'positive'
        elif polarity > -0.5:
            polarity_level = 'negative'
        else:
            polarity_level = 'very_negative'
        
        if abs(polarity) < 0.1:
            polarity_level = 'neutral'
        
        # Build context
        context = f"""<user_emotional_context>
Current User Emotional State:
- Sentiment: {polarity:.2f} ({'positive' if polarity > 0 else 'negative' if polarity < 0 else 'neutral'})
- Intensity: {intensity:.2f} ({'high' if intensity > 0.7 else 'moderate' if intensity > 0.3 else 'low'})
- Detected Emotion: {emotion}

Response Guidelines:
{self.EMOTION_GUIDELINES.get(emotion, self.EMOTION_GUIDELINES['neutral'])}
Please maintain a {self.POLARITY_GUIDELINES[polarity_level]} tone throughout your response.
</user_emotional_context>

"""
        
        # Optionally include embedding
        if self.include_embedding and 'emotion_embedding' in emotion_state:
            embedding_str = ', '.join([f"{x:.3f}" for x in emotion_state['emotion_embedding'][:5]])
            context += f"[Emotion Embedding (first 5 dims): {embedding_str}...]\n\n"
        
        return context
    
    def build_prompt(
        self,
        user_message: str,
        emotion_state: Optional[Dict[str, any]] = None,
        include_emotion: bool = True
    ) -> str:
        """
        Build complete prompt with emotion context
        
        Args:
            user_message: User's question or message
            emotion_state: Emotion state dictionary (optional)
            include_emotion: Whether to include emotion context
        
        Returns:
            Complete prompt string
        """
        if not include_emotion or emotion_state is None:
            # Return user message without emotion context
            return user_message
        
        # Build emotion-aware prompt
        emotion_context = self.build_emotion_context(emotion_state)
        
        prompt = f"""{emotion_context}User Question: {user_message}

Please respond to the user's question considering their emotional state described above."""
        
        return prompt
    
    def build_messages(
        self,
        messages: List[Dict[str, str]],
        emotion_state: Optional[Dict[str, any]] = None,
        include_emotion: bool = True
    ) -> List[Dict[str, str]]:
        """
        Build messages with emotion context for chat format
        
        Args:
            messages: List of message dicts [{"role": "user", "content": "..."}]
            emotion_state: Emotion state dictionary
            include_emotion: Whether to include emotion context
        
        Returns:
            Modified messages list with emotion context in first user message
        """
        if not include_emotion or emotion_state is None or len(messages) == 0:
            return messages
        
        # Find the last user message and prepend emotion context
        modified_messages = messages.copy()
        
        for i in range(len(modified_messages) - 1, -1, -1):
            if modified_messages[i].get('role') == 'user':
                original_content = modified_messages[i]['content']
                
                # Handle different content types (text or multimodal)
                if isinstance(original_content, str):
                    emotion_context = self.build_emotion_context(emotion_state)
                    modified_messages[i]['content'] = f"{emotion_context}{original_content}"
                elif isinstance(original_content, list):
                    # Multimodal content (text + images)
                    emotion_context = self.build_emotion_context(emotion_state)
                    # Insert emotion context before the first text element
                    for j, item in enumerate(original_content):
                        if item.get('type') == 'text':
                            original_content[j]['text'] = f"{emotion_context}{item['text']}"
                            break
                break
        
        return modified_messages


def format_emotion_display(emotion_state: Dict[str, any]) -> str:
    """
    Format emotion state for display in UI
    
    Args:
        emotion_state: Emotion state dictionary
    
    Returns:
        Formatted string for display
    """
    polarity = emotion_state['polarity']
    intensity = emotion_state['intensity']
    emotion = emotion_state['emotion_label']
    
    # Emoji mapping
    emoji_map = {
        'joy': '😊',
        'sadness': '😢',
        'anger': '😠',
        'fear': '😰',
        'surprise': '😲',
        'disgust': '🤢',
        'neutral': '😐'
    }
    
    emoji = emoji_map.get(emotion, '🙂')
    sentiment_emoji = '😊' if polarity > 0.3 else '😢' if polarity < -0.3 else '😐'
    
    display = f"""{emoji} **{emotion.upper()}**

📊 **Emotional Metrics:**
- Sentiment: {sentiment_emoji} {polarity:+.2f} ({'Positive' if polarity > 0 else 'Negative' if polarity < 0 else 'Neutral'})
- Intensity: {'🔥' * int(intensity * 5)} {intensity:.2%}

💡 **Response Tone:** {_get_tone_description(polarity, emotion)}
"""
    
    return display


def _get_tone_description(polarity: float, emotion: str) -> str:
    """Get description of recommended response tone"""
    if emotion == 'joy':
        return "Enthusiastic and celebratory"
    elif emotion == 'sadness':
        return "Empathetic and supportive"
    elif emotion == 'anger':
        return "Calm and solution-oriented"
    elif emotion == 'fear':
        return "Reassuring and clear"
    elif emotion == 'surprise':
        return "Informative and detailed"
    elif emotion == 'disgust':
        return "Constructive and alternative-focused"
    else:
        return "Balanced and informative"


if __name__ == "__main__":
    # Test prompt builder
    print("\n🧪 Testing EmotionAwarePromptBuilder\n")
    
    builder = EmotionAwarePromptBuilder()
    
    # Test case 1: Happy user
    print("="*60)
    print("Test 1: Happy User")
    print("="*60)
    emotion_state_happy = {
        'polarity': 0.8,
        'intensity': 0.9,
        'emotion_label': 'joy',
        'emotion_embedding': [0.1] * 32
    }
    
    prompt = builder.build_prompt(
        "I just got accepted to my dream university!",
        emotion_state_happy
    )
    print(prompt)
    
    # Test case 2: Frustrated user
    print("\n" + "="*60)
    print("Test 2: Frustrated User")
    print("="*60)
    emotion_state_angry = {
        'polarity': -0.6,
        'intensity': 0.7,
        'emotion_label': 'anger',
        'emotion_embedding': [0.1] * 32
    }
    
    prompt = builder.build_prompt(
        "Why isn't this code working? I've been trying for hours!",
        emotion_state_angry
    )
    print(prompt)
    
    # Test display formatting
    print("\n" + "="*60)
    print("Test 3: Display Formatting")
    print("="*60)
    display = format_emotion_display(emotion_state_happy)
    print(display)
    
    print("\n✅ Prompt builder tests completed!")
