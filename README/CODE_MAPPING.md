# 기존 코드에서 가져와야 할 부분 - 상세 매핑

## 📋 전체 흐름

```
┌─────────────────────────────────────────────────────────────┐
│                       CLIENT (On-Device)                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Step 1: 이미지/비디오 로드 & 전처리                          │
│  ┌────────────────────────────────────────────────────┐     │
│  │ qwen-vl-utils/vision_process.py                    │     │
│  │ - fetch_image()  (Line 95-147)                     │     │
│  │ - fetch_video()  (Line 405-478)                    │     │
│  │ - smart_resize() (Line 52-81)                      │     │
│  └────────────────────────────────────────────────────┘     │
│                         ↓                                    │
│  Step 2: Tokenization                                       │
│  ┌────────────────────────────────────────────────────┐     │
│  │ transformers.AutoProcessor                         │     │
│  │ - apply_chat_template()                            │     │
│  │ - tokenizer.encode()                               │     │
│  └────────────────────────────────────────────────────┘     │
│                         ↓                                    │
│  Step 3: Vision Encoder Forward                             │
│  ┌────────────────────────────────────────────────────┐     │
│  │ transformers.Qwen3VLForConditionalGeneration       │     │
│  │ → full_model.visual (Vision Encoder만 추출)        │     │
│  │   - patch_embed                                    │     │
│  │   - ViT blocks                                     │     │
│  │   - merger (DeepStack)                             │     │
│  └────────────────────────────────────────────────────┘     │
│                         ↓                                    │
│  Step 4: 데이터 전송                                         │
│  ┌────────────────────────────────────────────────────┐     │
│  │ - input_ids                                        │     │
│  │ - vision_embeddings                                │     │
│  │ - vision_token_positions                           │     │
│  │ - attention_mask                                   │     │
│  └────────────────────────────────────────────────────┘     │
└──────────────────────────┬──────────────────────────────────┘
                           │ Network (REST/gRPC)
┌──────────────────────────▼──────────────────────────────────┐
│                       SERVER (GPU)                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Step 5: Vision Embedding 삽입                               │
│  ┌────────────────────────────────────────────────────┐     │
│  │ - Text embeddings: language_model.embed_tokens()   │     │
│  │ - Vision embeddings 병합                           │     │
│  └────────────────────────────────────────────────────┘     │
│                         ↓                                    │
│  Step 6: LLM Prefill                                        │
│  ┌────────────────────────────────────────────────────┐     │
│  │ transformers.Qwen3VLForConditionalGeneration       │     │
│  │ → language_model.forward()                         │     │
│  │   - First forward pass                             │     │
│  │   - KV cache 생성                                  │     │
│  └────────────────────────────────────────────────────┘     │
│                         ↓                                    │
│  Step 7: Auto-regressive Decoding                           │
│  ┌────────────────────────────────────────────────────┐     │
│  │ - Next token prediction                            │     │
│  │ - KV cache 재사용                                  │     │
│  │ - Streaming response                               │     │
│  └────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 파일별 상세 매핑

### 1️⃣ Client - 전처리 코드

#### **파일: `qwen-vl-utils/src/qwen_vl_utils/vision_process.py`**

```python
# ============================================
# 이미지 전처리 함수들
# ============================================

# Line 39-50: 해상도 조정 유틸리티
def round_by_factor(number: int, factor: int) -> int:
    """가장 가까운 factor의 배수로 반올림"""
    return round(number / factor) * factor

def ceil_by_factor(number: int, factor: int) -> int:
    """factor의 배수로 올림"""
    return math.ceil(number / factor) * factor

def floor_by_factor(number: int, factor: int) -> int:
    """factor의 배수로 내림"""
    return math.floor(number / factor) * factor


# Line 52-81: 동적 해상도 조정 (핵심!)
def smart_resize(
    height: int, 
    width: int, 
    factor: int,  # Qwen3VL: 32, Qwen2.5VL: 28
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None
) -> Tuple[int, int]:
    """
    이미지를 다음 조건에 맞게 리사이즈:
    1. height, width 모두 factor의 배수
    2. 총 픽셀 수가 [min_pixels, max_pixels] 범위 내
    3. 종횡비 최대한 유지
    """
    # 구현 내용...
    return h_bar, w_bar


# Line 84-91: RGB 변환
def to_rgb(pil_image: Image.Image) -> Image.Image:
    """RGBA → RGB 변환 (투명 배경 → 흰색)"""
    if pil_image.mode == 'RGBA':
        white_background = Image.new("RGB", pil_image.size, (255, 255, 255))
        white_background.paste(pil_image, mask=pil_image.split()[3])
        return white_background
    else:
        return pil_image.convert("RGB")


# Line 95-147: 이미지 로드 & 전처리 (핵심!)
def fetch_image(
    ele: Dict[str, Union[str, Image.Image]], 
    image_patch_size: int = 14  # Qwen3VL: 16
) -> Image.Image:
    """
    이미지 로드 및 전처리
    - URL, local path, base64, PIL.Image 지원
    - 자동 리사이즈
    """
    # 1. 이미지 로드
    if isinstance(image, Image.Image):
        image_obj = image
    elif image.startswith("http://") or image.startswith("https://"):
        # URL에서 다운로드
        response = requests.get(image, stream=True)
        image_obj = Image.open(BytesIO(response.content))
    elif image.startswith("file://"):
        # 로컬 파일
        image_obj = Image.open(image[7:])
    elif image.startswith("data:image"):
        # Base64
        _, base64_data = image.split("base64,", 1)
        data = base64.b64decode(base64_data)
        image_obj = Image.open(BytesIO(data))
    else:
        # 기본 로컬 경로
        image_obj = Image.open(image)
    
    # 2. RGB 변환
    image = to_rgb(image_obj)
    
    # 3. 리사이즈
    patch_factor = int(image_patch_size * SPATIAL_MERGE_SIZE)  # 16*2=32
    
    if "resized_height" in ele and "resized_width" in ele:
        # 사용자 지정 크기
        resized_height, resized_width = smart_resize(
            ele["resized_height"], ele["resized_width"], factor=patch_factor
        )
    else:
        # min_pixels, max_pixels 기반 자동 조정
        width, height = image.size
        min_pixels = ele.get("min_pixels", IMAGE_MIN_TOKEN_NUM * patch_factor ** 2)
        max_pixels = ele.get("max_pixels", IMAGE_MAX_TOKEN_NUM * patch_factor ** 2)
        resized_height, resized_width = smart_resize(
            height, width, factor=patch_factor,
            min_pixels=min_pixels, max_pixels=max_pixels
        )
    
    image = image.resize((resized_width, resized_height))
    return image


# Line 150-184: 비디오 프레임 수 계산
def smart_nframes(
    ele: Dict[str, Any],
    total_frames: int,
    video_fps: Union[int, float]
) -> int:
    """
    비디오에서 샘플링할 프레임 수 계산
    - fps 기반 또는 nframes 직접 지정
    """
    if "nframes" in ele:
        nframes = round_by_factor(ele["nframes"], FRAME_FACTOR)
    else:
        fps = ele.get("fps", FPS)  # 기본 2.0 FPS
        min_frames = ceil_by_factor(ele.get("min_frames", FPS_MIN_FRAMES), FRAME_FACTOR)
        max_frames = floor_by_factor(ele.get("max_frames", FPS_MAX_FRAMES), FRAME_FACTOR)
        
        nframes = total_frames / video_fps * fps
        nframes = min(min(max(nframes, min_frames), max_frames), total_frames)
        nframes = floor_by_factor(nframes, FRAME_FACTOR)
    
    return nframes


# Line 187-226: Torchvision으로 비디오 읽기
def _read_video_torchvision(ele: Dict[str, Any]) -> Tuple[torch.Tensor, float]:
    """torchvision.io.read_video 사용"""
    video_path = ele["video"]
    
    video, audio, info = io.read_video(
        video_path,
        start_pts=ele.get("video_start", 0.0),
        end_pts=ele.get("video_end", None),
        pts_unit="sec",
        output_format="TCHW"
    )
    
    total_frames, video_fps = video.size(0), info["video_fps"]
    nframes = smart_nframes(ele, total_frames, video_fps)
    
    # 프레임 샘플링
    idx = torch.linspace(0, total_frames - 1, nframes).round().long()
    video = video[idx]
    
    return video, video_metadata, sample_fps


# Line 295-337: Decord로 비디오 읽기 (대안)
def _read_video_decord(ele: Dict[str, Any]) -> Tuple[torch.Tensor, float]:
    """decord.VideoReader 사용 (더 빠름)"""
    import decord
    vr = decord.VideoReader(ele["video"])
    
    total_frames, video_fps = len(vr), vr.get_avg_fps()
    nframes = smart_nframes(ele, total_frames, video_fps)
    
    idx = torch.linspace(0, total_frames - 1, nframes).round().long().tolist()
    video = vr.get_batch(idx).asnumpy()
    video = torch.tensor(video).permute(0, 3, 1, 2)
    
    return video, video_metadata, sample_fps


# Line 405-478: 비디오 통합 처리 (핵심!)
def fetch_video(
    ele: Dict[str, Any],
    image_patch_size: int = 14,
    return_video_metadata: bool = False
) -> Union[torch.Tensor, List[Image.Image]]:
    """
    비디오 로드 및 전처리
    - 프레임 샘플링
    - 리사이즈
    """
    # 1. 비디오 읽기 (backend 선택)
    video_reader_backend = get_video_reader_backend()
    video, video_metadata, sample_fps = VIDEO_READER_BACKENDS[video_reader_backend](ele)
    
    # 2. 프레임 리사이즈
    nframes, _, height, width = video.shape
    
    min_pixels = ele.get("min_pixels", VIDEO_FRAME_MIN_PIXELS)
    total_pixels = ele.get("total_pixels", MODEL_SEQ_LEN * image_factor ** 2 * 0.9)
    max_pixels = max(min(VIDEO_FRAME_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR), 
                     int(min_pixels * 1.05))
    
    resized_height, resized_width = smart_resize(
        height, width, factor=image_factor,
        min_pixels=min_pixels, max_pixels=max_pixels
    )
    
    # 3. Resize 적용
    video = transforms.functional.resize(
        video, [resized_height, resized_width],
        interpolation=InterpolationMode.BICUBIC, antialias=True
    ).float()
    
    return video, video_metadata


# Line 483-501: Vision 정보 추출
def extract_vision_info(
    conversations: Union[List[Dict], List[List[Dict]]]
) -> List[Dict[str, Any]]:
    """메시지에서 이미지/비디오 정보만 추출"""
    vision_infos = []
    
    for conversation in conversations:
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if ("image" in ele or "video" in ele or 
                        ele.get("type") in ("image", "video")):
                        vision_infos.append(ele)
    
    return vision_infos


# Line 508-534: 통합 전처리 함수 (핵심!)
def process_vision_info(
    conversations: List[Dict],
    return_video_kwargs: bool = False,
    return_video_metadata: bool = False,
    image_patch_size: int = 14
) -> Tuple[Optional[List[Image.Image]], Optional[List[torch.Tensor]], Optional[Dict]]:
    """
    모든 vision 정보 처리
    - 이미지 & 비디오 로드
    - 전처리
    """
    vision_infos = extract_vision_info(conversations)
    
    image_inputs = []
    video_inputs = []
    
    for vision_info in vision_infos:
        if "image" in vision_info or "image_url" in vision_info:
            image_inputs.append(fetch_image(vision_info, image_patch_size))
        elif "video" in vision_info:
            video_input, video_metadata, sample_fps = fetch_video(
                vision_info, image_patch_size, return_video_metadata=True
            )
            video_inputs.append((video_input, video_metadata))
    
    return image_inputs, video_inputs, video_kwargs
```

**Client에서 필요한 것:**
- ✅ 위 함수들 전체 복사
- ✅ Constants (Line 1-37)
- ✅ 의존성: PIL, torch, torchvision, requests

---

### 2️⃣ Client - Tokenization

#### **파일: transformers 라이브러리**

```python
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-235B-A22B-Instruct")

# ============================================
# 사용할 메서드들
# ============================================

# 1. Chat template 적용
text = processor.apply_chat_template(
    messages,
    tokenize=False,  # 먼저 텍스트로 변환
    add_generation_prompt=True
)
# 출력: "<|im_start|>user\n<image>Describe this image.<|im_end|>\n<|im_start|>assistant\n"

# 2. Tokenization
encoded = processor.tokenizer.encode(text, return_tensors="pt")
# 출력: input_ids = [1, 2, 3, ..., 151655, ...]  # 151655 = <image> token

# 3. 통합 처리 (이미지 포함)
inputs = processor(
    text=text,
    images=images,  # PIL.Image 리스트
    videos=videos,  # torch.Tensor
    do_resize=False,  # qwen-vl-utils에서 이미 처리함
    return_tensors="pt"
)
# 출력:
# {
#     'input_ids': tensor([[1, 2, 3, ..., 151655, ...]]),
#     'attention_mask': tensor([[1, 1, 1, ..., 1]]),
#     'pixel_values': tensor([...]),  # [num_images, C, H, W]
#     'image_grid_thw': tensor([[1, 9, 13]]),  # [num_images, 3]
# }
```

**Client에서 필요한 것:**
- ✅ `pip install transformers`
- ✅ Processor만 로드 (모델 불필요)

---

### 3️⃣ Client - Vision Encoder

#### **파일: transformers.models.qwen3_vl.modeling_qwen3_vl**

```python
from transformers import Qwen3VLForConditionalGeneration

# ============================================
# Vision Encoder 추출
# ============================================

# 전체 모델 로드
full_model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-235B-A22B-Instruct",
    torch_dtype="auto",
    device_map="cpu"  # 또는 "cuda"
)

# Vision Encoder만 추출
vision_encoder = full_model.visual  # Qwen3VLVisionModel 객체

# 모델 구조:
# vision_encoder
# ├── patch_embed: PatchEmbed
# ├── blocks: nn.ModuleList (ViT transformer blocks)
# │   ├── block[0]: VisionTransformerBlock
# │   ├── block[1]: VisionTransformerBlock
# │   └── ...
# └── merger: Qwen3VLMerger (DeepStack)


# ============================================
# Vision Encoder Forward
# ============================================

import torch

@torch.no_grad()
def encode_vision(pixel_values, image_grid_thw):
    """
    Args:
        pixel_values: torch.Tensor [batch, channels, height, width]
        image_grid_thw: torch.Tensor [num_images, 3] - (T, H, W)
    
    Returns:
        vision_outputs: torch.Tensor [num_patches, hidden_dim]
    """
    vision_outputs = vision_encoder(
        pixel_values=pixel_values,
        grid_thw=image_grid_thw
    )
    
    return vision_outputs


# ============================================
# 내부 동작 (참고용)
# ============================================

class Qwen3VLVisionModel(nn.Module):
    def forward(self, pixel_values, grid_thw):
        # 1. Patch Embedding
        x = self.patch_embed(pixel_values)  # [B, num_patches, embed_dim]
        
        # 2. ViT Blocks
        hidden_states = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in self.merger.layer_indices:  # DeepStack용
                hidden_states.append(x)
        
        # 3. DeepStack Merger (multi-level feature fusion)
        merged_features = self.merger(hidden_states)
        
        return merged_features  # [num_patches, hidden_dim]
```

**Client에서 필요한 것:**
- ✅ `full_model.visual` 추출
- ✅ 메모리: Vision Encoder만 로드 (~2-3GB for 32B model의 vision part)
- ⚠️ **문제점**: Vision Encoder도 크기가 있어서 On-Device에 부담될 수 있음

---

### 4️⃣ Server - LLM 전용

#### **파일: transformers.models.qwen3_vl.modeling_qwen3_vl**

```python
from transformers import Qwen3VLForConditionalGeneration

# ============================================
# Server에서 Vision Embeddings 받아서 LLM만 실행
# ============================================

class ServerLLMInference:
    def __init__(self, model_name):
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            attn_implementation="flash_attention_2"
        )
        self.model.eval()
    
    @torch.no_grad()
    def generate_from_vision_embeddings(
        self,
        input_ids,  # [batch, seq_len]
        vision_embeddings,  # [num_patches, hidden_dim]
        vision_token_positions,  # <image> 토큰 위치
        attention_mask=None,
        max_new_tokens=128
    ):
        """
        Vision embeddings을 받아서 LLM만 실행
        """
        # 1. Text embeddings
        text_embeds = self.model.language_model.embed_tokens(input_ids)
        # text_embeds shape: [batch, seq_len, hidden_dim]
        
        # 2. Vision embeddings 삽입
        inputs_embeds = text_embeds.clone()
        
        vision_idx = 0
        for batch_idx in range(input_ids.shape[0]):
            for pos in vision_token_positions[batch_idx]:
                if input_ids[batch_idx, pos] == 151655:  # <image> token
                    # Vision embedding으로 대체
                    inputs_embeds[batch_idx, pos] = vision_embeddings[vision_idx]
                    vision_idx += 1
        
        # 3. LLM Forward (Prefill + Decoding)
        outputs = self.model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            do_sample=True,
            temperature=0.7
        )
        
        return outputs


# ============================================
# 기존 방식과 비교
# ============================================

# 기존 (Server에서 전체 처리):
outputs = model.generate(
    input_ids=input_ids,
    pixel_values=pixel_values,  # Server에서 Vision Encoder 실행
    image_grid_thw=image_grid_thw,
    max_new_tokens=128
)

# 새로운 방식 (Client에서 Vision Encoder 실행):
outputs = model.language_model.generate(
    inputs_embeds=merged_embeddings,  # Vision + Text
    max_new_tokens=128
)
```

**Server에서 필요한 것:**
- ✅ `language_model.generate()` 사용
- ✅ Vision Encoder 건너뛰기
- ✅ Vision embeddings 직접 삽입

---

## 🎯 핵심 포인트

### Client에서 가져와야 할 코드:

1. **전처리 (필수):**
   - ✅ `qwen-vl-utils/src/qwen_vl_utils/vision_process.py` 전체
   - ✅ `transformers.AutoProcessor`

2. **Vision Encoder (선택):**
   - ✅ `full_model.visual` 추출
   - ⚠️ 크기: ~2-3GB (32B 모델 기준)
   - 💡 **대안**: Server에서 실행하고 pixel_values만 전송

3. **통신:**
   - ✅ requests 또는 gRPC
   - ✅ JSON/Protobuf 직렬화

### Server에서 수정할 코드:

1. **Vision Embedding 수신:**
   - ✅ 새로운 입력 파라미터 추가: `vision_embeddings`
   - ✅ Vision Encoder 건너뛰기

2. **LLM만 실행:**
   - ✅ `language_model.generate(inputs_embeds=...)`
   - ✅ KV cache 활용

---

## 다음 단계

실제 구현 코드를 원하시면 다음 중 선택해주세요:

1. **Option A: Client 전처리 + Server (Vision Encoder + LLM)**
   - 가장 간단
   - 네트워크 대역폭 적음

2. **Option B: Client (전처리 + Vision Encoder) + Server LLM**
   - 균형잡힌 접근
   - Vision Encoder도 On-Device

3. **Option C: Full Client Preprocessing + Server Inference Only**
   - Server 부하 최소화
   - Client에 Vision Encoder 필요

어떤 옵션으로 진행할까요?
