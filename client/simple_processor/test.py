from raw_process import RawProcess
from pipeline import Pipeline
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(base_dir, "../../examples", "sad_woman.mp4")
data_path = os.path.join(base_dir, "processed_data")
 
data_name = "sample_video"
device = 'cuda'

pipeline = Pipeline(device, data_path, data_name)
result = pipeline(video_path)

print("Pipeline output:", result.shape) # emotion
import ipdb; ipdb.set_trace()