# MedGemma 1.5 环境安装完成 ✅

## 安装总结

### 已完成
✅ **创建conda环境** medgemma
- Python 3.11.14
- 环境路径: `/Users/lizhanbing12/miniconda3/envs/medgemma`
- 磁盘占用: 1.6GB

✅ **安装核心依赖**
- PyTorch 2.9.1 (conda-forge，MPS支持)
- Transformers 4.57.5
- Accelerate 1.12.0

✅ **安装工具库**
- Pillow 12.1.0
- Requests 2.32.5
- Hugging Face Hub 0.36.0
- Librosa 0.11.0

✅ **验证环境**
- MPS加速: ✅ 可用
- 环境隔离: ✅ 完全独立
- 磁盘空间: ✅ 257GB可用

---

## ⚠️ 需要额外步骤：Hugging Face访问权限

MedGemma 1.5是受保护的模型，需要：
1. 接受Hugging Face的使用条款
2. 设置Hugging Face访问令牌

### 步骤1: 接受条款
1. 访问: https://huggingface.co/google/medgemma-1.5-4b-it
2. 点击 "Agree and access repository" 按钮
3. 登录你的Hugging Face账号（没有则注册）

### 步骤2: 创建访问令牌
1. 访问: https://huggingface.co/settings/tokens
2. 点击 "New token"
3. 命名为: medgemma_access
4. 类型选择: Read
5. 复制生成的token

### 步骤3: 配置token
```bash
# 激活medgemma环境
conda activate medgemma

# 使用huggingface-cli登录
huggingface-cli login

# 粘贴你的token
Paste your token here: hf_xxxxxxxxxxxxxx
```

---

## 📋 使用MedGemma 1.5

### Python脚本
```python
from transformers import pipeline
import torch

# 创建pipeline
pipe = pipeline(
    "text-generation",
    model="google/medgemma-1.5-4b-it",
    torch_dtype=torch.bfloat16,
    device="mps",  # Mac M2加速
)

# 使用模型
messages = [
    {
        "role": "user",
        "content": "如何诊断细菌性肺炎？"
    }
]

output = pipe(text=messages, max_new_tokens=300)
print(output[0]["generated_text"][-1]["content"])
```

### 运行测试脚本
```bash
conda activate medgemma
python test_medgemma.py
```

---

## 🎯 下一步

### 选项1: 安装MedASR（医学语音识别）
```bash
conda activate medgemma
pip install git+https://github.com/huggingface/transformers.git
```

### 选项2: 门诊记录生成系统
需要以下组件：
1. ESP32-S3音频采集
2. MedASR语音转文字
3. MedGemma 1.5生成门诊记录

---

## 🔧 环境管理

### 激活环境
```bash
conda activate medgemma
```

### 退出环境
```bash
conda deactivate
```

### 查看环境列表
```bash
conda env list
```

### 删除环境（如需要）
```bash
conda env remove -n medgemma
```

---

## 📞 遇到问题？

### Hugging Face登录失败
```bash
# 手动设置token
export HF_TOKEN=hf_xxxxxxxxxxxxxx

# 或在代码中使用
from huggingface_hub import login
login(token="hf_xxxxxxxxxxxxxx")
```

### 模型下载慢
```bash
# 设置镜像加速
export HF_ENDPOINT=https://hf-mirror.com
```

### MPS不可用
```bash
# 检查PyTorch版本
python -c "import torch; print(torch.__version__)"

# 如需要，重新安装
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio
```

---

## 📚 参考资料

- [MedGemma官方文档](https://developers.google.com/health-ai-developer-foundations/medgemma)
- [Hugging Face模型页](https://huggingface.co/google/medgemma-1.5-4b-it)
- [Transformers文档](https://huggingface.co/docs/transformers)
- [PyTorch MPS支持](https://pytorch.org/docs/stable/notes/mps.html)

---

## ✅ 安装完成！

medgemma环境已成功创建并配置。在完成Hugging Face访问权限设置后，你就可以开始使用MedGemma 1.5进行医学文本生成了。
