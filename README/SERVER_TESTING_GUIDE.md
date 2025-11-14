# Server Testing Guide

Server 코드를 Client에서 생성한 vision embedding으로 테스트하는 가이드입니다.

## 📋 테스트 절차

### **Phase 1: Client에서 Embedding 생성**

```bash
cd client

# 1. 이미지 embedding 생성
python save_test_embeddings.py image <path_to_image> ../test_data

# 2. 비디오 embedding 생성
python save_test_embeddings.py video <path_to_video> ../test_data

# 예시:
python save_test_embeddings.py image ../cookbooks/assets/omni_recognition/image_example.jpg
python save_test_embeddings.py video ../cookbooks/assets/omni_recognition/video_example.mp4
```

**생성되는 파일:**
- `{media_type}_{filename}_tensors.pt` - PyTorch 텐서 (input_ids, vision_embeddings, etc.)
- `{media_type}_{filename}_metadata.json` - 메타데이터
- `{media_type}_{filename}_info.txt` - 사람이 읽을 수 있는 요약

---

### **Phase 2: Server에서 Embedding으로 테스트**

```bash
cd server

# 1. 의존성 설치 (처음 한 번만)
pip install -r requirements.txt

# 2. 저장된 embedding으로 테스트
python test_server_with_embeddings.py ../test_data/image_example_tensors.pt
python test_server_with_embeddings.py ../test_data/video_example_tensors.pt
```

**테스트 내용:**
- ✅ Embedding 로드
- ✅ Server LLM 초기화
- ✅ Non-streaming 생성
- ✅ Streaming 생성

---

### **자동화 스크립트 (옵션)**

전체 테스트를 한 번에 실행:

```bash
# 프로젝트 루트에서
./run_tests.sh
```

이 스크립트는:
1. Client에서 이미지/비디오 embedding 생성
2. Server에서 각 embedding 테스트
3. 결과 요약 출력

---

## 📁 저장된 데이터 구조

```
test_data/
├── image_example_tensors.pt       # 이미지 텐서
├── image_example_metadata.json    # 이미지 메타데이터
├── image_example_info.txt         # 이미지 정보
├── video_example_tensors.pt       # 비디오 텐서
├── video_example_metadata.json    # 비디오 메타데이터
└── video_example_info.txt         # 비디오 정보
```

### **텐서 파일 내용**

```python
data = torch.load('image_example_tensors.pt')
# Keys:
# - 'input_ids': torch.Tensor, shape [1, seq_len]
# - 'attention_mask': torch.Tensor, shape [1, seq_len]
# - 'vision_embeddings': torch.Tensor, shape [num_patches, hidden_dim]
# - 'vision_token_positions': List[int]
```

---

## 🧪 수동 테스트 예시

### **Python에서 직접 로드**

```python
import torch

# 1. 데이터 로드
data = torch.load('test_data/image_example_tensors.pt')

# 2. Server 초기화
from llm_inference import ServerLLMInference
llm = ServerLLMInference(model_name="Qwen/Qwen3-VL-2B-Instruct")

# 3. 생성
response = llm.generate(
    input_ids=data['input_ids'],
    vision_embeddings=data['vision_embeddings'],
    vision_token_positions=data['vision_token_positions'],
    attention_mask=data['attention_mask'],
    max_new_tokens=128
)

print(response)
```

---

## 🔍 디버깅

### **문제: Embedding 파일이 생성되지 않음**

```bash
# Client 환경 확인
cd client
python -c "import torch; from preprocessor import ClientPreprocessor; print('OK')"

# GPU 사용 가능 확인
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### **문제: Server 테스트 실패**

```bash
# Server 환경 확인
cd server
python -c "from llm_inference import ServerLLMInference; print('OK')"

# Embedding shape 확인
python -c "import torch; d = torch.load('../test_data/image_example_tensors.pt'); print(d['vision_embeddings'].shape)"
```

---

## 📊 예상 출력

### **save_test_embeddings.py**

```
================================================================================
💾 Saving IMAGE Embedding Data for Server Testing
================================================================================

[Step 1/5] Initializing preprocessor and encoder...
📦 Loading processor for Qwen/Qwen3-VL-2B-Instruct...
✅ Processor loaded! Patch size: 16
🚀 Loading Vision Encoder from Qwen/Qwen3-VL-2B-Instruct...
   Target device: cuda:0
   ✅ Vision Encoder loaded!

[Step 2/5] Creating messages for image...

[Step 3/5] Preprocessing...
   ✅ Preprocessing complete:
      - input_ids: torch.Size([1, 45])
      - attention_mask: torch.Size([1, 45])

[Step 4/5] Encoding vision features...
🎨 Encoding vision features...
   Input pixel_values shape: torch.Size([1225, 1176])
   Grid THW shape: torch.Size([1, 3]), values: tensor([[1, 35, 35]])
✅ Vision encoding complete!
   Output shape: torch.Size([1225, 2048])
   ✅ Vision embeddings: torch.Size([1225, 2048])
      - Vision token positions: [9]

[Step 5/5] Saving data...
   ✅ Saved: ../test_data/image_example_tensors.pt
   ✅ Saved: ../test_data/image_example_metadata.json
   ✅ Saved: ../test_data/image_example_info.txt

🎉 All data saved successfully!
```

### **test_server_with_embeddings.py**

```
================================================================================
🧪 Testing Server with Pre-computed Embeddings
================================================================================

[Step 1/4] Loading embeddings from ../test_data/image_example_tensors.pt...
   ✅ Data loaded:
      - input_ids: torch.Size([1, 45])
      - vision_embeddings: torch.Size([1225, 2048])
      - vision_token_positions: [9]

[Step 2/4] Initializing Server LLM...
🚀 Starting Qwen3-VL Inference Server
✅ Model loaded on cuda:0
   ✅ Server LLM loaded!

[Step 3/4] Generating response (non-streaming)...
🔥 Starting generation...
🚀 Running LLM generation...
✅ Generation complete: 245 chars

================================================================================
📝 Generated Response:
================================================================================
이 이미지는 해변의 일몰 장면을 보여줍니다. 하늘은 주황색과 분홍색으로 물들어 있으며...
================================================================================

[Step 4/4] Testing streaming generation...
================================================================================
📝 Streaming Response:
================================================================================
이 이미지는 해변의 일몰 장면을 보여줍니다...
================================================================================

✅ Streaming generation complete!

📊 Test Summary
✅ All server tests passed!
```

---

## 🚀 다음 단계

Embedding 테스트가 성공하면:

1. **FastAPI 서버 시작**
   ```bash
   cd server
   python server_api.py
   ```

2. **Gradio Client 시작**
   ```bash
   cd client
   python gradio_app.py
   ```

3. **End-to-End 테스트**
   - 브라우저에서 `http://localhost:7860` 접속
   - 이미지/비디오 업로드
   - 텍스트 프롬프트 입력
   - 생성 결과 확인
