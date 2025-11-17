## Architecture Diagram
```markdown
┌─────────────────────────────────────────────────────────┐
│              Client Container (Gradio UI)               │
│                                                         │
│  ┌───────────────────────────────────────────────────┐  │
│  │         Gradio Web UI (Port 7860)                 │  │
│  │  - Text input (question)                          │  │
│  │  - Image/Video upload (question content)          │  │
│  │  - Webcam/Mic access (real-time emotion capture)  │  │
│  └───────────────────────────────────────────────────┘  │
│                         ↓                               │
│  ════════════════════════════════════════════════════   │
│   STREAM 1: User Emotion (Real-time)                    │
│  ════════════════════════════════════════════════════   │
│                         ↓                               │
│  ┌───────────────────────────────────────────────────┐  │
│  │  Webcam Stream → Face/Head Pose Detection         │  │
│  └───────────────────────────────────────────────────┘  │
│                         ↓                               │
│  ┌───────────────────────────────────────────────────┐  │
│  │  Microphone → Voice/Prosody Analysis              │  │
│  └───────────────────────────────────────────────────┘  │
│                         ↓                               │
│  ┌───────────────────────────────────────────────────┐  │
│  │  Emotion Vision Encoder (Small ViT, On-device)    │  │
│  │  Audio Feature Extractor (pitch/energy/prosody)   │  │
│  └───────────────────────────────────────────────────┘  │
│                         ↓                               │
│  ┌───────────────────────────────────────────────────┐  │
│  │  UniMSE Multimodal Fusion Encoder                 │  │
│  │   - Fuse text + visual + audio                    │  │
│  │   - Contrastive learning embedding                │  │
│  │   - Output: {polarity, intensity, emotion_label}  │  │
│  └───────────────────────────────────────────────────┘  │
│                         ↓                               │
│              emotion_state (JSON + embedding)           │
│                                                         │
│  ════════════════════════════════════════════════════   │
│   STREAM 2: Question Content (Image/Video)              │
│  ════════════════════════════════════════════════════   │
│                         ↓                               │
│  ┌───────────────────────────────────────────────────┐  │
│  │  Vision Preprocessing (qwen-vl-utils)             │  │
│  └───────────────────────────────────────────────────┘  │
│                         ↓                               │
│  ┌───────────────────────────────────────────────────┐  │
│  │  Content Vision Encoder (Large ViT, 2-3GB)        │  │
│  │  * Only for question image/video                  │  │
│  │  → content_vision_emb (2048-dim)                  │  │
│  └───────────────────────────────────────────────────┘  │
│                         ↓                               │
│  ════════════════════════════════════════════════════   │
│   SEND TO SERVER                                        │
│  ════════════════════════════════════════════════════   │
│                         ↓                               │
│  ┌───────────────────────────────────────────────────┐  │
│  │  HTTP POST /api/v1/generate                       │  │
│  │  Payload:                                         │  │
│  │   - user_question_text                            │  │
│  │   - content_vision_emb (question image info)      │  │
│  │   - emotion_state (real-time user emotion)        │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────┼───────────────────────────────────────┘
                  │ HTTP POST
                  │ /api/v1/generate
                  ↓
┌─────────────────────────────────────────────────────────┐
│                  Server Container                       │
│                                                         │
│  ┌───────────────────────────────────────────────────┐  │
│  │       FastAPI Server (Port 8001)                  │  │
│  └───────────────────────────────────────────────────┘  │
│                         ↓                               │
│  ┌───────────────────────────────────────────────────┐  │
│  │     Emotion-aware Prompt Builder                  │  │
│  │  - user_question_text                             │  │
│  │  - emotion_state (polarity/intensity/emotion)     │  │
│  │  - Adjust tone / empathy / safety                 │  │
│  └───────────────────────────────────────────────────┘  │
│                         ↓                               │
│  ┌───────────────────────────────────────────────────┐  │
│  │       LLM Inference (GPU Accelerated)             │  │
│  │  - Integrate content_vision_emb (understand Q)    │  │
│  │  - Apply emotion_state (adjust response style)    │  │
│  └───────────────────────────────────────────────────┘  │
│                         ↓                               │
│            Enhanced Chatbot Response                    │
│         (emotion-aware, context-rich)                   │
└─────────────────────────────────────────────────────────┘
```

## Novelty
1. **Emotion-aware Multimodal Interaction**: Real-time emotion capture via webcam and microphone, integrated with text and visual inputs for a more empathetic chatbot response.

2. **On-Device First Architecture**: Lightweight emotion encoders run on the client side, reducing latency and preserving user privacy by minimizing data sent to the server.
    - 절대 RAW 영상/음성 데이터를 서버로 전송하지 않음

3. **Advanced Multimodal Fusion**: Utilization of UniMSE for effective fusion of text, visual, and audio modalities to accurately interpret user emotions.


## Methodology
1. 감정 인식 품질

2. 다운스트림 챗봇 품질

3. 시스템 리소스 효율성
