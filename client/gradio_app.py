"""
Gradio Web UI for Qwen3-VL Client
Provides user interface for multimodal chat with emotion recognition
"""

import gradio as gr
import os
import torch
from preprocessor import ClientPreprocessor
from vision_encoder import ClientVisionEncoder
from client_api import ClientAPI
from simple_processor_wrapper import SimpleProcessorWrapper, format_emotion_for_prompt, format_emotion_display
from typing import List, Tuple, Optional, Dict
import logging
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure Gradio's logger to use the same format
gradio_logger = logging.getLogger("gradio")
gradio_logger.setLevel(logging.INFO)

class Qwen3VLClient:
    """Main client application integrating all components"""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-2B-Instruct",
        server_url: str = "http://server:8000",
        use_vision_encoder: bool = True,
        enable_emotion: bool = True
    ):
        """
        Initialize Qwen3-VL Client
        
        Args:
            model_name: Model to use for preprocessing
            server_url: URL of the inference server
            use_vision_encoder: Whether to run vision encoder on client
            enable_emotion: Whether to enable emotion recognition
        """
        logger.info("=" * 60)
        logger.info("🚀 Initializing Qwen3-VL Client")
        logger.info("=" * 60)
        
        # Initialize preprocessor
        self.preprocessor = ClientPreprocessor(model_name)
        
        # Initialize vision encoder (optional)
        self.use_vision_encoder = use_vision_encoder
        if use_vision_encoder:
            self.vision_encoder = ClientVisionEncoder(model_name)
        else:
            self.vision_encoder = None
            logger.warning("⚠️  Vision Encoder disabled (will send pixel values to server)")
        
        # Initialize emotion recognition (simple_processor)
        self.enable_emotion = enable_emotion
        if enable_emotion:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.emotion_processor = SimpleProcessorWrapper(device=device)
            logger.info("✅ Emotion recognition enabled (simple_processor)")
        else:
            self.emotion_processor = None
            logger.info("ℹ️  Emotion recognition disabled")
        
        # Initialize API client
        self.api_client = ClientAPI(server_url)
        
        # Check server health
        if self.api_client.check_health():
            logger.info("✅ Server is healthy!")
        else:
            logger.warning("⚠️  Server health check failed!")
        
        logger.info("=" * 60)
    
    def extract_emotion_from_video(
        self, 
        emotion_video_path: str,
        text: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Extract emotion state from user's emotion video using simple_processor
        
        Args:
            emotion_video_path: Path to video file with user's face/voice
            text: User's text input (not used by simple_processor)
            
        Returns:
            Emotion result dictionary or None if emotion disabled
        """
        if not self.enable_emotion or emotion_video_path is None:
            return None
        
        try:
            logger.info(f"🎭 Extracting emotion from video (simple_processor)")
            
            # Extract emotion using simple_processor
            emotion_result = self.emotion_processor.extract_emotion_from_video(emotion_video_path)
            
            logger.info(f"✅ Detected emotion: {emotion_result['dominant_emotion']} "
                       f"(sentiment={emotion_result['sentiment']:.3f})")
            
            return emotion_result
            
        except Exception as e:
            logger.error(f"❌ Emotion extraction failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def process_and_generate(
        self,
        messages: List[dict],
        emotion_result: Optional[Dict] = None,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        stream: bool = False
    ):
        """
        Process messages and generate response
        
        Args:
            messages: List of message dictionaries
            emotion_result: Optional emotion result from simple_processor
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stream: Whether to stream response
            
        Returns:
            Generated text (or iterator if streaming)
        """
        # Inject emotion context into messages if available
        if emotion_result is not None:
            # Format emotion for prompt
            emotion_text = format_emotion_for_prompt(emotion_result)
            
            # Log emotion injection for debugging
            logger.info("🎭 Injecting emotion context into prompt:")
            logger.info(f"   Dominant: {emotion_result['dominant_emotion']}")
            logger.info(f"   Sentiment: {emotion_result['sentiment']:.3f}")
            
            # Find last user message and prepend emotion context
            for i in range(len(messages) - 1, -1, -1):
                if messages[i].get('role') == 'user':
                    content = messages[i]['content']
                    if isinstance(content, list):
                        # Multimodal content - add to first text element
                        for item in content:
                            if item.get('type') == 'text':
                                original_text = item['text']
                                item['text'] = emotion_text + item['text']
                                logger.info(f"   ✅ Emotion context added to multimodal prompt")
                                logger.info(f"   Original: {original_text[:50]}...")
                                logger.info(f"   With emotion: {item['text'][:200]}...")
                                break
                    elif isinstance(content, str):
                        # Text-only content
                        original_content = content
                        messages[i]['content'] = emotion_text + content
                        logger.info(f"   ✅ Emotion context added to text-only prompt")
                        logger.info(f"   Original: {original_content[:50]}...")
                        logger.info(f"   With emotion: {messages[i]['content'][:200]}...")
                    break
        
        # 1. Preprocess
        inputs = self.preprocessor.preprocess(messages)
        
        # 2. Encode vision (if enabled on client)
        vision_embeddings = None
        vision_positions = []
        
        if self.use_vision_encoder:
            # Check for either pixel_values (images) or pixel_values_videos (videos)
            # Text-only case: no vision data
            if 'pixel_values_videos' in inputs:
                pixel_values = inputs['pixel_values_videos']
                # logger.info(f"🎬 Processing video with shape: {pixel_values.shape}")
                vision_embeddings = self.vision_encoder.encode(
                    pixel_values,
                    inputs.get('image_grid_thw'),
                    inputs.get('video_grid_thw')
                )
                # Get vision token positions
                vision_positions = self.preprocessor.extract_vision_token_positions(
                    inputs['input_ids']
                )
            elif 'pixel_values' in inputs:
                pixel_values = inputs['pixel_values']
                # logger.info(f"📷 Processing image with shape: {pixel_values.shape}")
                vision_embeddings = self.vision_encoder.encode(
                    pixel_values,
                    inputs.get('image_grid_thw'),
                    inputs.get('video_grid_thw')
                )
                # Get vision token positions
                vision_positions = self.preprocessor.extract_vision_token_positions(
                    inputs['input_ids']
                )
            else:
                # Text-only input (no image/video)
                logger.info("💬 Text-only input (no vision data)")
                vision_embeddings = None
                vision_positions = []
        else:
            raise NotImplementedError(
                "Vision encoder is disabled. "
                "Sending pixel_values to server is not yet implemented. "
                "Set USE_VISION_ENCODER=true to enable client-side encoding."
            )
        
        # 3. Generate on server
        if stream:
            return self.api_client.generate_stream(
                inputs['input_ids'],
                vision_embeddings,
                vision_positions,
                inputs['attention_mask'],
                max_new_tokens,
                temperature
            )
        else:
            return self.api_client.generate(
                inputs['input_ids'],
                vision_embeddings,
                vision_positions,
                inputs['attention_mask'],
                max_new_tokens,
                temperature
            )


def create_gradio_interface(client: Qwen3VLClient):
    """Create Gradio interface for the client"""
    
    def chat_fn(message, history, image, video, emotion_video, enable_emotion_checkbox,max_tokens_value, temperature_value):
        """Process chat message with optional image/video and emotion"""
        
        # Extract emotion result if emotion video provided and enabled
        emotion_result = None
        emotion_display_text = None
        
        if client.enable_emotion and enable_emotion_checkbox and emotion_video is not None:
            try:
                # Extract emotion from video
                emotion_result = client.extract_emotion_from_video(
                    emotion_video_path=emotion_video,
                    text=message  # text parameter kept for future use
                )
                if emotion_result:
                    emotion_display_text = format_emotion_display(emotion_result)
                    # Add debug info to display
                    emotion_display_text += "\n\n---\n**🔍 Debug Info:**\n"
                    emotion_display_text += f"Emotion context will be injected into your prompt.\n"
                    emotion_display_text += f"Check terminal logs for full prompt details."
                else:
                    emotion_display_text = "⚠️ Failed to detect emotion. Please try another video."
            except Exception as e:
                logger.error(f"❌ Emotion extraction failed: {e}")
                emotion_display_text = f"❌ Error: {str(e)}"
        else:
            emotion_display_text = "🎭 **Emotion Recognition**\n\nUpload a video and enable emotion recognition to get emotion-aware responses."
        
        # Build messages list following the same format as test_vision_embedding.py
        # Note: process_vision_info expects direct keys without "type" for images/videos
        messages = []
        content = []
        
        # Add image if provided (use absolute path without "type" key)
        if image is not None:
            import os
            image_abs_path = os.path.abspath(image)
            content.append({"image": image_abs_path})  # Direct key, no "type"
        
        # Add video if provided (use absolute path without "type" key)
        if video is not None:
            import os
            video_abs_path = os.path.abspath(video)
            content.append({"video": video_abs_path})  # Direct key, no "type"
        
        # Add text (text DOES need "type" key)
        if message:
            content.append({"type": "text", "text": message})
        
        messages.append({"role": "user", "content": content})
        
        # Generate response (streaming)
        try:
            full_response = ""
            # Build history format: [(user_msg, bot_msg), ...]
            # Append current user message
            new_history = history + [(message, "")]
            
            for partial_text in client.process_and_generate(
                messages,
                emotion_result=emotion_result,  # Pass emotion result
                max_new_tokens=max_tokens_value,
                temperature=temperature_value,
                stream=True
            ):
                full_response = partial_text
                # Update last message with streaming response
                new_history[-1] = (message, full_response)
                yield new_history, emotion_display_text
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            new_history = history + [(message, error_msg)]
            yield new_history, emotion_display_text
    
    # Create interface
    with gr.Blocks(title="Qwen3-VL Client") as demo:
        gr.Markdown("""
        # 🤖 Qwen3-VL Client with Emotion Recognition
        
        Emotion-Aware Multimodal AI Assistant powered by Qwen3-VL + Simple Processor
        
        **Client-Server Architecture:**
        - 🖥️ Client: Vision preprocessing, encoding & emotion recognition (7D emotion vector)
        - 🚀 Server: LLM inference with emotion-aware prompting
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📸 Question Content")
                image_input = gr.Image(
                    type="filepath",
                    label="📷 Upload Image (for your question)"
                )
                video_input = gr.Video(
                    label="🎬 Upload Video (for your question)"
                )
                
                gr.Markdown("### 🎭 Your Emotion State")
                enable_emotion = gr.Checkbox(
                    label="Enable Emotion Recognition",
                    value=client.enable_emotion,
                    interactive=True
                )
                gr.Markdown("Upload a video of yourself (with face & voice) for emotion-aware responses")
                emotion_video_input = gr.Video(
                    label="🎥 Your Video"
                )
                
                emotion_display = gr.Markdown(
                    "🎭 **Emotion Recognition**\n\n"
                    "Upload a video and click Send to analyze your emotion state.\n\n"
                    "The emotion will be detected when you submit your message."
                )
                
                gr.Markdown("### Generation Settings")
                max_tokens = gr.Slider(
                    minimum=32, maximum=512, value=128, step=32,
                    label="Max New Tokens"
                )
                temperature = gr.Slider(
                    minimum=0.1, maximum=1.5, value=0.7, step=0.1,
                    label="Temperature"
                )
            
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=600
                )
                msg = gr.Textbox(
                    label="Your Message",
                    placeholder="Ask about the image/video...",
                    lines=2
                )
                with gr.Row():
                    submit = gr.Button("🚀 Send", variant="primary")
                    clear = gr.Button("🧹 Clear")
        
        # Event handlers
        submit.click(
            chat_fn,
            inputs=[msg, chatbot, image_input, video_input, emotion_video_input, enable_emotion,max_tokens, temperature],
            outputs=[chatbot, emotion_display]
        )
        
        msg.submit(
            chat_fn,
            inputs=[msg, chatbot, image_input, video_input, emotion_video_input, enable_emotion,max_tokens,temperature],
            outputs=[chatbot, emotion_display]
        )
        
        clear.click(
            lambda: (None, None, None, None, []),
            outputs=[image_input, video_input, emotion_video_input, msg, chatbot]
        )
        
        # Clear message box after sending
        submit.click(lambda: "", None, msg)
        msg.submit(lambda: "", None, msg)
    
    return demo


def main():
    """Main entry point"""
    # Configuration from environment variables
    model_name = os.getenv("MODEL_NAME", "Qwen/Qwen3-VL-2B-Instruct")
    server_url = os.getenv("SERVER_URL", "http://server:8001")
    use_vision_encoder = os.getenv("USE_VISION_ENCODER", "true").lower() == "true"
    enable_emotion = os.getenv("ENABLE_EMOTION", "true").lower() == "true"
    gradio_server_name = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    gradio_server_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    
    logger.info(f"Configuration: model={model_name}, server={server_url}, "
               f"use_vision_encoder={use_vision_encoder}, enable_emotion={enable_emotion}")
    
    # Initialize client
    client = Qwen3VLClient(
        model_name=model_name,
        server_url=server_url,
        use_vision_encoder=use_vision_encoder,
        enable_emotion=enable_emotion
    )
    
    # Create and launch Gradio interface
    logger.info(f"🚀 Launching Gradio on {gradio_server_name}:{gradio_server_port}")
    demo = create_gradio_interface(client)
    demo.launch(
        server_name=gradio_server_name,
        server_port=gradio_server_port,
        share=True
    )


if __name__ == "__main__":
    main()
