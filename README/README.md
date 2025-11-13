# Client-Server 분리 아키텍처 구현 가이드

## 개요
Client에서 전처리와 Vision Encoder를 실행하고, Server에서는 LLM Prefill/Decoding만 수행하도록 분리합니다.

## 필요한 코드 위치

### 1. Client Side (On-Device) - 전처리 코드

#### 📁 `qwen-vl-utils/src/qwen_vl_utils/vision_process.py`
Client에서 이미지/비디오 전처리에 필요한 모든 함수가 포함되어 있습니다.

**가져와야 할 주요 함수들:**

```python
# 1. 전처리 유틸리티 함수들
- smart_resize(height, width, factor, min_pixels, max_pixels)  # 동적 해상도 조정
- to_rgb(pil_image)  # RGB 변환
- round_by_factor(), ceil_by_factor(), floor_by_factor()  # 해상도 계산

# 2. 이미지 처리
- fetch_image(ele, image_patch_size)  # 이미지 로드 & 리사이즈

# 3. 비디오 처리
- smart_nframes(ele, total_frames, video_fps)  # 프레임 수 계산
- calculate_video_frame_range()  # 프레임 범위 계산
- _read_video_torchvision(ele)  # torchvision으로 비디오 읽기
- _read_video_decord(ele)  # decord로 비디오 읽기 (선택)
- _read_video_torchcodec(ele)  # torchcodec로 비디오 읽기 (선택)
- fetch_video(ele, image_patch_size)  # 비디오 로드 & 리사이즈

# 4. 통합 함수
- extract_vision_info(conversations)  # 메시지에서 vision 정보 추출
- process_vision_info(conversations, image_patch_size)  # 전체 vision 처리
```

**파일 위치:** `/home/shjung/Qwen3-VL-Agent/qwen-vl-utils/src/qwen_vl_utils/vision_process.py`

---

#### 📁 `transformers` 라이브러리의 Processor
Client에서 tokenization에 필요합니다.

```python
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-235B-A22B-Instruct")

# 필요한 메서드:
- processor.apply_chat_template()  # 메시지 → 텍스트 변환
- processor.tokenizer.encode()  # 텍스트 → input_ids
- processor.image_processor  # 이미지 전처리 설정
- processor(text, images, videos)  # 통합 전처리
```

---

### 2. Vision Encoder 코드

Vision Encoder는 **transformers 라이브러리**에 구현되어 있습니다.

#### 🔍 Vision Encoder 모델 구조

```python
# transformers.models.qwen3_vl.modeling_qwen3_vl.py (transformers 라이브러리 내부)

class Qwen3VLVisionModel:
    """Vision Encoder (ViT)"""
    def __init__(self, config):
        self.patch_embed = ...  # Patch embedding
        self.blocks = nn.ModuleList([...])  # ViT transformer blocks
        self.merger = Qwen3VLMerger(config)  # DeepStack
    
    def forward(self, pixel_values, grid_thw):
        # 1. Patch embedding
        x = self.patch_embed(pixel_values)
        
        # 2. ViT blocks
        for block in self.blocks:
            x = block(x)
        
        # 3. DeepStack merger (multi-level feature fusion)
        vision_outputs = self.merger(x)
        
        return vision_outputs


class Qwen3VLForConditionalGeneration:
    """전체 모델"""
    def __init__(self, config):
        self.visual = Qwen3VLVisionModel(config)  # Vision Encoder
        self.language_model = Qwen3Model(config)  # LLM
        self.lm_head = nn.Linear(...)  # Output head
```

#### 📦 Vision Encoder만 추출하는 방법

```python
from transformers import Qwen3VLForConditionalGeneration

# 전체 모델 로드
full_model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-235B-A22B-Instruct"
)

# Vision Encoder만 추출
vision_encoder = full_model.visual  # Qwen3VLVisionModel

# Vision Encoder Forward
vision_outputs = vision_encoder(
    pixel_values=pixel_values,  # [batch, channels, height, width]
    grid_thw=image_grid_thw  # [num_images, 3] - (T, H, W)
)
# 출력: vision_outputs.shape = [num_patches, hidden_dim]
```

---

### 3. Server Side - LLM만 필요

Server에서는 Vision Encoder의 출력(vision embeddings)을 받아서 LLM만 실행합니다.

#### 필요한 코드:

```python
# transformers.models.qwen3_vl.modeling_qwen3_vl.py

class Qwen3VLForConditionalGeneration:
    def forward(
        self,
        input_ids,
        attention_mask,
        pixel_values=None,  # Client에서 처리하면 불필요
        image_grid_thw=None,  # Client에서 처리하면 불필요
        vision_embeddings=None,  # Client에서 전달받음 (새로 추가 필요)
        **kwargs
    ):
        # Vision Encoder 건너뛰고 바로 LLM으로
        if vision_embeddings is not None:
            # Client에서 받은 vision embeddings 사용
            inputs_embeds = self._merge_vision_embeddings(
                input_ids, vision_embeddings
            )
        else:
            # 기존 방식 (Server에서 Vision Encoder 실행)
            vision_outputs = self.visual(pixel_values, image_grid_thw)
            inputs_embeds = self._merge_vision_embeddings(
                input_ids, vision_outputs
            )
        
        # LLM Forward
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        
        return outputs
```

---

## 구체적인 구현 단계

### Step 1: Client 전처리 모듈 생성

```python
# client_preprocessor.py

from qwen_vl_utils import (
    extract_vision_info,
    fetch_image,
    fetch_video,
    process_vision_info,
    smart_resize
)
from transformers import AutoProcessor
import torch

class ClientPreprocessor:
    def __init__(self, model_name):
        self.processor = AutoProcessor.from_pretrained(model_name)
        
    def preprocess(self, messages):
        """Client에서 전체 전처리 수행"""
        
        # 1. Vision 정보 처리
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        images, videos, video_kwargs = process_vision_info(
            messages,
            image_patch_size=self.processor.image_processor.patch_size,
            return_video_kwargs=True,
            return_video_metadata=True
        )
        
        # 2. Tokenization
        inputs = self.processor(
            text=text,
            images=images,
            videos=videos,
            do_resize=False,
            return_tensors="pt",
            **video_kwargs
        )
        
        return inputs
```

**필요한 파일:**
- ✅ `qwen-vl-utils/src/qwen_vl_utils/vision_process.py` (전체 복사)
- ✅ `qwen-vl-utils/src/qwen_vl_utils/__init__.py`

---

### Step 2: Client Vision Encoder 실행

```python
# client_vision_encoder.py

from transformers import Qwen3VLForConditionalGeneration
import torch

class ClientVisionEncoder:
    def __init__(self, model_name, device='cuda'):
        # 전체 모델 로드 (Vision Encoder만 사용)
        full_model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map=device
        )
        
        # Vision Encoder만 추출
        self.vision_encoder = full_model.visual
        self.vision_encoder.eval()
        
    @torch.no_grad()
    def encode(self, pixel_values, image_grid_thw, video_grid_thw=None):
        """Vision Encoder Forward"""
        
        # Vision Encoder 실행
        vision_outputs = self.vision_encoder(
            pixel_values=pixel_values,
            grid_thw=torch.cat([image_grid_thw, video_grid_thw]) if video_grid_thw else image_grid_thw
        )
        
        return vision_outputs  # [num_patches, hidden_dim]
```

**필요한 코드:**
- ✅ `transformers` 라이브러리 (pip install transformers)
- ✅ Vision Encoder 부분만 추출 (full_model.visual)

---

### Step 3: Server LLM 전용 모듈

```python
# server_llm_inference.py

from transformers import Qwen3VLForConditionalGeneration
import torch

class ServerLLMInference:
    def __init__(self, model_name):
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            attn_implementation="flash_attention_2"
        )
        self.model.eval()
        
    def generate_from_embeddings(
        self,
        input_ids,
        vision_embeddings,
        vision_token_positions,
        attention_mask=None,
        max_new_tokens=128
    ):
        """Vision embeddings을 받아서 LLM만 실행"""
        
        # Text embeddings
        text_embeds = self.model.language_model.embed_tokens(input_ids)
        
        # Vision embeddings 삽입
        inputs_embeds = self._merge_embeddings(
            text_embeds, vision_embeddings, vision_token_positions
        )
        
        # LLM Generation
        with torch.no_grad():
            outputs = self.model.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                use_cache=True
            )
        
        return outputs
    
    def _merge_embeddings(self, text_embeds, vision_embeds, positions):
        """Text와 Vision embedding 결합"""
        # positions: <image> 토큰이 있는 위치
        # vision_embeds를 해당 위치에 삽입
        
        for i, pos in enumerate(positions):
            text_embeds[:, pos] = vision_embeds[i]
        
        return text_embeds
```

---

### Step 4: 통신 프로토콜

```python
# API 설계

# Client → Server
{
    "input_ids": [1, 2, 3, ..., 151655, ...],  # 151655 = <image> token
    "vision_embeddings": [[...], [...], ...],  # [num_patches, hidden_dim]
    "vision_token_positions": [10, 11, 12, ...],  # <image> 토큰 위치
    "attention_mask": [1, 1, 1, ...],
    "max_new_tokens": 128
}

# Server → Client
{
    "generated_ids": [42, 15, 89, ...],
    "text": "Generated response..."
}
```

---

## 파일 구조

```
client_inference/
├── README.md (이 파일)
├── client/
│   ├── preprocessor.py        # Step 1: 전처리
│   ├── vision_encoder.py      # Step 2: Vision Encoder
│   ├── client_api.py          # Client API
│   └── requirements.txt
│       - qwen-vl-utils
│       - transformers
│       - torch
│       - Pillow
│       - requests
│
└── server/
    ├── llm_inference.py       # Step 3: LLM 전용
    ├── server_api.py          # FastAPI Server
    └── requirements.txt
        - transformers
        - torch
        - fastapi
        - uvicorn
```

---

## 다음 단계

실제 코드 구현을 원하시면 말씀해주세요. 다음을 생성해드릴 수 있습니다:

1. ✅ `client/preprocessor.py` - qwen-vl-utils 기반 전처리
2. ✅ `client/vision_encoder.py` - Vision Encoder 추출
3. ✅ `client/client_api.py` - Client 통합 API
4. ✅ `server/llm_inference.py` - Server LLM 전용
5. ✅ `server/server_api.py` - FastAPI 서버
6. ✅ 예제 코드

어떤 부분부터 구현할까요?
