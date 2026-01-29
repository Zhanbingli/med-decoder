import torch
from transformers import pipeline
import huggingface_hub

# 1. 强制使用 float32 排除精度导致的输出为空问题
device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.float32
print(f"🚀 运行设备: {device} | 精度: {dtype}")

model_id = "google/medasr"

# 2. 加载 Pipeline
pipe = pipeline(
    "automatic-speech-recognition",
    model=model_id,
    device=device,
    torch_dtype=dtype,
    trust_remote_code=True
)

# 3. 下载音频
audio_path = huggingface_hub.hf_hub_download(model_id, 'test_audio.wav')

# 4. 推理
print("正在识别（使用 float32 稳定性更高）...")
result = pipe(
    audio_path,
    chunk_length_s=20,
    stride_length_s=2,
    batch_size=1
)

print("\n" + "="*30)
print("医学识别结果：")
print(f"[{result['text']}]") # 加上方括号看看到底是不是空
print("="*30)
