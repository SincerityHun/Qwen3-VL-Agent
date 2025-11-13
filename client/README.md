
## Dev
```bash
uv venv -p 3.12
source .venv/bin/activate
uv pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install -r requirements.txt

# FOR TEST VISION ENCODER in Client
python test_vision_embedding.py image ../examples/dog.jpg
python test_vision_embedding.py video ../examples/IronMan.mp4

# FOR 

```


## Deploy