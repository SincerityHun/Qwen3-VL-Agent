"""
UniMSE Emotion Recognition Wrapper
Wraps the original UniMSE model without modification
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, List, Union
import logging

# Add unimse_src to path
sys.path.insert(0, str(Path(__file__).parent / 'unimse_src'))

from transformers import T5Tokenizer

logger = logging.getLogger(__name__)


class UniMSEConfig:
    """UniMSE configuration matching the original config.py"""
    def __init__(self):
        # Multimodal settings
        self.multi = True
        self.adapter_name = 'ffn'  # 'ffn', 'parallel', or 'cross-atten'
        self.use_adapter = True
        self.use_prefix_p = False
        self.info_nce = False
        self.add_va = False
        self.visualize = False
        self.adapter_layer = 3
        
        # Visual encoder settings
        self.d_vin = 35  # Visual feature dimension
        self.d_vh = 32   # Visual hidden dimension
        self.d_vout = 32 # Visual output dimension
        self.dropout_v = 0.2
        
        # Acoustic encoder settings
        self.d_ain = 74  # Acoustic feature dimension
        self.d_ah = 32   # Acoustic hidden dimension
        self.d_aout = 32 # Acoustic output dimension
        self.dropout_a = 0.2
        
        # RNN settings
        self.n_layer = 1
        self.bidirectional = False
        
        # Text encoder settings (T5)
        self.d_tin = 512   # T5-small hidden size
        self.d_tout = 512
        self.init_checkpoint = None  # Path to pretrained T5 checkpoint
        self.fine_T5 = False  # Whether to fine-tune T5 (False for inference)
        self.fine_T5_layers = ['block.10', 'block.11']
        
        # Contrastive learning
        self.use_cl = False
        self.cl_name = 'info_nce'
        self.use_info_nce_num = 3
        
        # P-tuning v2
        self.pre_seq_len = 8
        self.prompt_hidden_size = 64
        self.prefix_hidden_size = 64
        self.num_hidden_layers = 1
        self.prefix_projection = False
        self.hidden_dropout_prob = 0.3
        
        # Other settings
        self.s_dim = 30
        self.fuse = True
        self.pred_type = 'generation'  # 'regression', 'classification', or 'generation'
        
        # Training settings (not used for inference)
        self.batch_size = 1
        self.gradient_accumulation_step = 1
        self.hidden_size = 512
        

class UniMSEEmotionWrapper:
    """
    Wrapper for UniMSE model that provides emotion recognition interface
    without modifying the original UniMSE code
    """
    
    EMOTION_LABELS = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']
    
    def __init__(
        self, 
        checkpoint_path: Optional[str] = None,
        device: str = 'cpu',
        adapter_name: str = 'ffn'
    ):
        """
        Initialize UniMSE emotion wrapper
        
        Args:
            checkpoint_path: Path to trained UniMSE checkpoint (optional)
            device: Device to run model on ('cpu' or 'cuda')
            adapter_name: Adapter type ('ffn', 'parallel', or 'cross-atten')
        """
        self.device = device
        self.checkpoint_path = checkpoint_path
        
        # Create config
        self.hp = UniMSEConfig()
        self.hp.adapter_name = adapter_name
        
        logger.info(f"🎭 Initializing UniMSE Emotion Wrapper")
        logger.info(f"   Device: {device}")
        logger.info(f"   Adapter: {adapter_name}")
        
        # Initialize T5 tokenizer
        logger.info("📝 Loading T5 tokenizer...")
        self.tokenizer = T5Tokenizer.from_pretrained('t5-small')
        
        # Initialize UniMSE model (original code, no modification)
        logger.info("🔧 Loading UniMSE model...")
        self._init_model()
        
        # Load checkpoint if provided
        if checkpoint_path and Path(checkpoint_path).exists():
            logger.info(f"📥 Loading checkpoint: {checkpoint_path}")
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        else:
            logger.warning("⚠️  No checkpoint loaded - using random initialization")
            logger.warning("   Predictions will be random until model is trained")
        
        self.model.eval()
        logger.info("✅ UniMSE wrapper initialized")
    
    def _init_model(self):
        """Initialize UniMSE model using original code"""
        from unimse_src.model import Model
        from unimse_src.config import DEVICE
        
        # Override DEVICE to use our device
        import unimse_src.config as config_module
        config_module.DEVICE = torch.device(self.device)
        
        # Create model (now uses 't5-small' from modified encoders.py)
        self.model = Model(self.hp).to(self.device)
    
    def encode(
        self,
        text: Optional[Union[str, List[str]]] = None,
        visual_features: Optional[torch.Tensor] = None,
        visual_lengths: Optional[torch.Tensor] = None,
        acoustic_features: Optional[torch.Tensor] = None,
        acoustic_lengths: Optional[torch.Tensor] = None
    ) -> Dict:
        """
        Encode multimodal features into emotion state
        
        Args:
            text: Text input (string or list of strings)
            visual_features: (seq_len, 35) or (batch, seq_len, 35)
            visual_lengths: (1,) or (batch,)
            acoustic_features: (seq_len, 74) or (batch, seq_len, 74)
            acoustic_lengths: (1,) or (batch,)
            
        Returns:
            Emotion state dictionary with:
                - emotion_label: str
                - emotion_index: int
                - polarity: float
                - intensity: float
        """
        # Validate inputs
        if visual_features is None or acoustic_features is None:
            raise ValueError("Both visual and acoustic features are required")
        
        # Handle text input
        if text is None or text == "":
            text = "neutral"  # Default text for UniMSE
        
        # Convert single string to list
        if isinstance(text, str):
            text = [text]
        
        # Prepare text inputs for T5
        text_inputs = self.tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        )
        
        t5_input_id = text_inputs['input_ids'].to(self.device)
        t5_att_mask = text_inputs['attention_mask'].to(self.device)
        
        # Ensure batch dimension for visual/acoustic features
        if visual_features.dim() == 2:
            visual_features = visual_features.unsqueeze(0)
        if acoustic_features.dim() == 2:
            acoustic_features = acoustic_features.unsqueeze(0)
        if visual_lengths.dim() == 0:
            visual_lengths = visual_lengths.unsqueeze(0)
        if acoustic_lengths.dim() == 0:
            acoustic_lengths = acoustic_lengths.unsqueeze(0)
        
        # Move to device
        visual_features = visual_features.to(self.device)
        acoustic_features = acoustic_features.to(self.device)
        visual_lengths = visual_lengths.to(self.device)
        acoustic_lengths = acoustic_lengths.to(self.device)
        
        # Call UniMSE model (use forward instead of generate for simplicity)
        with torch.no_grad():
            # Create dummy labels (not used for inference, but required by forward)
            dummy_labels = t5_input_id.clone()
            dummy_labels[dummy_labels == self.tokenizer.pad_token_id] = -100
            
            # Forward pass
            model_output = self.model.forward(
                sentences=text,
                t5_input_id=t5_input_id,
                t5_att_mask=t5_att_mask,
                t5_labels=dummy_labels,
                ids=[0] * t5_input_id.size(0),  # Dummy IDs
                visual=visual_features,
                acoustic=acoustic_features,
                v_len=visual_lengths,
                a_len=acoustic_lengths
            )
            
            logits, loss = model_output
            
            # Get predictions from logits (argmax)
            pred_ids = torch.argmax(logits, dim=-1)
        
        # Decode output
        generated_text = self.tokenizer.decode(pred_ids[0], skip_special_tokens=True)
        
        # Parse emotion from generated text
        emotion_state = self._parse_emotion_output(generated_text)
        
        return emotion_state
    
    def _parse_emotion_output(self, generated_text: str) -> Dict:
        """
        Parse UniMSE's generated text output into emotion state
        
        Args:
            generated_text: Text generated by UniMSE T5 model
            
        Returns:
            Emotion state dictionary
        """
        text = generated_text.lower().strip()
        
        # Sentiment mapping (for MOSI/MOSEI datasets)
        sentiment_map = {
            'positive': 0.8,
            'very positive': 0.9,
            'negative': -0.8,
            'very negative': -0.9,
            'neutral': 0.0,
            'weakly positive': 0.3,
            'weakly negative': -0.3,
        }
        
        # Emotion mapping (for IEMOCAP/MELD datasets)
        emotion_map = {
            'anger': 0, 'angry': 0,
            'disgust': 1, 'disgusted': 1,
            'fear': 2, 'fearful': 2, 'scared': 2,
            'happiness': 3, 'happy': 3, 'joy': 3,
            'sadness': 4, 'sad': 4,
            'surprise': 5, 'surprised': 5,
            'neutral': 6,
        }
        
        # Try to match sentiment first
        if text in sentiment_map:
            polarity = sentiment_map[text]
            # Map sentiment to emotion
            if polarity > 0.5:
                emotion_label = 'happiness'
            elif polarity < -0.5:
                emotion_label = 'sadness'
            else:
                emotion_label = 'neutral'
        
        # Try to match emotion
        elif text in emotion_map:
            emotion_label = text.replace('angry', 'anger').replace('happy', 'happiness').replace('sad', 'sadness').replace('scared', 'fear')
            # Map emotion to polarity
            if emotion_label in ['happiness', 'surprise']:
                polarity = 0.6
            elif emotion_label in ['anger', 'disgust', 'fear', 'sadness']:
                polarity = -0.6
            else:
                polarity = 0.0
        
        # Default fallback
        else:
            logger.warning(f"⚠️  Unknown emotion output: '{text}' - defaulting to neutral")
            emotion_label = 'neutral'
            polarity = 0.0
        
        # Normalize emotion label
        if emotion_label not in self.EMOTION_LABELS:
            # Find closest match
            for key in emotion_map:
                if key in emotion_label:
                    emotion_label = list(emotion_map.keys())[list(emotion_map.values()).index(emotion_map[key])]
                    break
        
        # Ensure valid emotion label
        if emotion_label not in self.EMOTION_LABELS:
            emotion_label = 'neutral'
        
        emotion_index = self.EMOTION_LABELS.index(emotion_label)
        intensity = abs(polarity)
        
        return {
            'emotion_label': emotion_label,
            'emotion_index': emotion_index,
            'polarity': float(polarity),
            'intensity': float(intensity),
            'raw_output': generated_text
        }


if __name__ == "__main__":
    # Simple test
    print("\n🧪 Testing UniMSE Emotion Wrapper")
    
    wrapper = UniMSEEmotionWrapper(device='cpu')
    
    # Create dummy features
    visual_features = torch.randn(10, 35)
    visual_lengths = torch.tensor([10])
    acoustic_features = torch.randn(20, 74)
    acoustic_lengths = torch.tensor([20])
    
    # Test encoding
    result = wrapper.encode(
        text="I am very happy today!",
        visual_features=visual_features,
        visual_lengths=visual_lengths,
        acoustic_features=acoustic_features,
        acoustic_lengths=acoustic_lengths
    )
    
    print(f"\n✅ Emotion: {result['emotion_label']}")
    print(f"   Polarity: {result['polarity']:.3f}")
    print(f"   Intensity: {result['intensity']:.3f}")
    print(f"   Raw output: {result['raw_output']}")
