## 1. Title
Emotion-Aware Multimodal LLM Serving Architecture
(Using On-Device Vision Encoder + UniMSE for Real-Time Affect Conditioning)

## 2. System Overview
본 아키텍처는 사용자의 감정 상태(MSA)·감정 분류(ERC)를 실시간으로 분석하여,
LLM의 응답을 사용자 친화적으로 조정하기 위한 Emotion-Aware Multimodal Serving Framework이다.

핵심 아이디어는 다음과 같다.

a) On-device Vision Encoder(예: Qwen-VL ViT)가 이미지/비디오 데이터를 전처리 및 임베딩해 Privacy를 보호한다. 원본 영상은 서버에 전송되지 않는다.

b) UniMSE 모델이 사용자의 실시간 음성·표정 등 멀티모달 신호를 기반으로 감정 상태(MSA/ERA)를 추출한다.

c) 감정 분석 결과를 Text Input에 부가 정보로 삽입하여 서버의 Qwen-VL LLM이 Emotion-aware 응답을 생성하도록 유도한다.

    
## 3. Architecture Diagram
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
### 3.1. Client Part
클라이언트는 크게 다음 세 부분으로 동작한다.
#### 3.1.1. 이미지/비디오 입력 처리 경로 (질문 관련 영상 입력)
```markdown
Image/Video for question input  →  PreProcessor  →  ViT(Qwen-VL)  →  Vision Embedding
```
1.  PreProcessor

    - 이미지/비디오 프레임을 Vision Encoder가 요구하는 해상도·정규화 포맷으로 변환
        
        e.g. Resize, Center-Crop, Normalization

2. Vision Encoder (Qwen-VL ViT)

    - on-device에서 실행

    - 입력된 이미지/비디오를 Token Sequence로 임베딩

    - 결과: Vision Embedding

#### 3.1.2. 텍스트 입력 처리 (질문 텍스트 입력)
```markdown
Text Input  →  Text Raw Data (수정 없이 LLM으로 전달)
```
- 사용자의 질문 또는 대화 입력

- 필요한 경우, 감정 정보를 주입하기 위한 placeholder 역할 수행

#### 3.1.3. 실시간 사용자 상태(표정·음성) 기반 감정 분석 경로
이 경로는 질문 입력과 독립적으로 실시간 Stream으로 동작한다.
```markdown
User Video + User Audio → Buffer (t_prev ~ t_cur) → UniMSE → MSA & ERA 결과
```
1. User Video, User Audio
    - 실시간 스트림 입력
    - t_prev ~ t_cur 구간의 단기 윈도우로 묶어 처리

2. Buffer

3. UniMSE (Multimodal Sentiment + Emotion Recognition)

출력:
- MSA: Multi-modal Sentiment Analysis
    
    e.g. positive, 1.6

- ERA: Emotion Recognition in Conversation

    e.g. joy


### 3.2. 감정 정보의 LLM Conditioning 방식
UniMSE 결과는 LLM에 직접 입력되는 것이 아니라 텍스트 입력에 soft-conditioning 형태로 주입한다.

e.g.
```yml
<user_context>
User Emotion:
 - Sentiment: positive(1.6)
 - Emotion: joy
</user_context>
```

이렇게 “감정 특성”을 Text Input과 함께 서버에 전송하면 서버 LLM이 답변 생성 시 tone/style/prioritization을 조절할 수 있다.

### 3.3. Server Part (Qwen-VL 기반 LLM)
```markdown
Client → { Vision Embedding + User Text + Emotion Tags } → Server LLM
```
- Qwen-VL 기반 멀티모달 LLM
    - Client에서 생성된 Vision Embedding을 그대로 받아 multimodal reasoning 수행
    - UniMSE에서 추출한 감정 정보를 conditioning 정보로 활용하여 Emotion-Adaptive Response 생성

- Output
    - 일반 텍스트 답변
    - 필요시 Follow-up 질문 생성
    - 감정에 맞는 대응(공감, 위로, 축하 등)

## Metrics for Evaluation
1. 감정 인식 품질

2. 다운스트림 챗봇 품질

3. 시스템 리소스 효율성

## Novelty of proposed System
1. **Real-time Emotion-aware Multimodal Interaction**: Real-time emotion capture via webcam and microphone, integrated with text and visual inputs for a more empathetic chatbot response.

2. **On-Device First Architecture(privacy-aware)**: Lightweight emotion encoders run on the client side, reducing latency and preserving user privacy by minimizing data sent to the server.
    - 절대 RAW 영상/음성 데이터를 서버로 전송하지 않음

3. **Resource Efficiency**: 

4. **Modular Design**:
