"""
Gradio Web UI for Qwen3-VL Client
Provides user interface for multimodal chat
"""

import gradio as gr
import os
import torch
from preprocessor import ClientPreprocessor
from vision_encoder import ClientVisionEncoder
from client_api import ClientAPI
from typing import List, Tuple, Optional


class Qwen3VLClient:
    """Main client application integrating all components"""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-2B-Instruct",
        server_url: str = "http://server:8000",
        use_vision_encoder: bool = True
    ):
        """
        Initialize Qwen3-VL Client
        
        Args:
            model_name: Model to use for preprocessing
            server_url: URL of the inference server
            use_vision_encoder: Whether to run vision encoder on client
        """
        print("=" * 60)
        print("🚀 Initializing Qwen3-VL Client")
        print("=" * 60)
        
        # Initialize preprocessor
        self.preprocessor = ClientPreprocessor(model_name)
        
        # Initialize vision encoder (optional)
        self.use_vision_encoder = use_vision_encoder
        if use_vision_encoder:
            self.vision_encoder = ClientVisionEncoder(model_name)
        else:
            self.vision_encoder = None
            print("⚠️  Vision Encoder disabled (will send pixel values to server)")
        
        # Initialize API client
        self.api_client = ClientAPI(server_url)
        
        # Check server health
        if self.api_client.check_health():
            print("✅ Server is healthy!")
        else:
            print("⚠️  Server health check failed!")
        
        print("=" * 60)
    
    def process_and_generate(
        self,
        messages: List[dict],
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        stream: bool = False
    ):
        """
        Process messages and generate response
        
        Args:
            messages: List of message dictionaries
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stream: Whether to stream response
            
        Returns:
            Generated text (or iterator if streaming)
        """
        # 1. Preprocess
        inputs = self.preprocessor.preprocess(messages)
        
        # 2. Encode vision (if enabled on client)
        if self.use_vision_encoder:
            # Check for either pixel_values (images) or pixel_values_videos (videos)
            # Following the same logic as test_vision_embedding.py
            if 'pixel_values_videos' in inputs:
                pixel_values = inputs['pixel_values_videos']
                print(f"🎬 Processing video with shape: {pixel_values.shape}")
            elif 'pixel_values' in inputs:
                pixel_values = inputs['pixel_values']
                print(f"📷 Processing image with shape: {pixel_values.shape}")
            else:
                raise ValueError("No pixel_values or pixel_values_videos found in preprocessed inputs")
            
            vision_embeddings = self.vision_encoder.encode(
                pixel_values,
                inputs.get('image_grid_thw'),
                inputs.get('video_grid_thw')
            )
        else:
            raise NotImplementedError(
                "Vision encoder is disabled. "
                "Sending pixel_values to server is not yet implemented. "
                "Set USE_VISION_ENCODER=true to enable client-side encoding."
            )
        
        # 3. Get vision token positions
        vision_positions = self.preprocessor.extract_vision_token_positions(
            inputs['input_ids']
        )
        
        # 4. Generate on server
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
    
    def chat_fn(message, history, image, video):
        """Process chat message with optional image/video"""
        
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
            for partial_text in client.process_and_generate(
                messages, 
                max_new_tokens=256,
                stream=True
            ):
                full_response = partial_text
                yield full_response
        except Exception as e:
            yield f"❌ Error: {str(e)}"
    
    # Create interface
    with gr.Blocks(title="Qwen3-VL Client") as demo:
        gr.Markdown("""
        # 🤖 Qwen3-VL Client
        
        Multimodal AI Assistant powered by Qwen3-VL
        
        **Client-Server Architecture:**
        - 🖥️ Client: Vision preprocessing & encoding
        - 🚀 Server: LLM inference
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    type="filepath",
                    label="📷 Upload Image"
                )
                video_input = gr.Video(
                    label="🎬 Upload Video"
                )
                
                gr.Markdown("### Generation Settings")
                # max_tokens = gr.Slider(
                #     minimum=32, maximum=512, value=128, step=32,
                #     label="Max New Tokens"
                # )
                # temperature = gr.Slider(
                #     minimum=0.1, maximum=1.5, value=0.7, step=0.1,
                #     label="Temperature"
                # )
            
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=500
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
            inputs=[msg, chatbot, image_input, video_input],
            outputs=[chatbot]
        )
        
        msg.submit(
            chat_fn,
            inputs=[msg, chatbot, image_input, video_input],
            outputs=[chatbot]
        )
        
        clear.click(
            lambda: (None, None, None, []),
            outputs=[image_input, video_input, msg, chatbot]
        )
    
    return demo


def main():
    """Main entry point"""
    # Configuration from environment variables
    model_name = os.getenv("MODEL_NAME", "Qwen/Qwen3-VL-2B-Instruct")
    server_url = os.getenv("SERVER_URL", "http://server:8000")
    use_vision_encoder = os.getenv("USE_VISION_ENCODER", "true").lower() == "true"
    gradio_server_name = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    gradio_server_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    
    # Initialize client
    client = Qwen3VLClient(
        model_name=model_name,
        server_url=server_url,
        use_vision_encoder=use_vision_encoder
    )
    
    # Create and launch Gradio interface
    demo = create_gradio_interface(client)
    demo.launch(
        server_name=gradio_server_name,
        server_port=gradio_server_port,
        share=True
    )


if __name__ == "__main__":
    main()
