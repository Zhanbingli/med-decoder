"""
CardioVoice Unified Model Manager
=================================

统一管理 MedASR (Transformers) 和 MedGemma (Ollama GGUF) 两个模型，
实现语音转文字和门诊记录生成的一体化服务。

Architecture:
    - MedASR: Transformers + PyTorch (MPS加速)
    - MedGemma: Ollama API + GGUF量化模型 (Metal加速)

Author: CardioVoice Team
Date: January 2026
Version: 1.0.0
"""

import os
import sys
import time
import json
import logging
import threading
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Union, Callable, Tuple
from enum import Enum
from abc import ABC, abstractmethod

import numpy as np
import torch
from transformers import pipeline
import ollama

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """模型加载状态"""
    UNINITIALIZED = "uninitialized"
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"
    UNLOADED = "unloaded"


class InferenceBackend(Enum):
    """推理后端类型"""
    TRANSFORMERS = "transformers"
    OLLAMA = "ollama"


@dataclass
class ModelInfo:
    """模型信息"""
    name: str
    version: str
    backend: InferenceBackend
    status: ModelStatus = ModelStatus.UNINITIALIZED
    load_time: float = 0.0
    memory_usage_mb: float = 0.0
    last_inference_time: float = 0.0
    inference_count: int = 0
    error_message: Optional[str] = None


@dataclass
class PatientInfo:
    """患者信息"""
    name: str
    age: int
    gender: str
    medical_history: Optional[str] = None
    medications: Optional[str] = None
    allergies: Optional[str] = None


@dataclass
class TranscriptionResult:
    """转写结果"""
    text: str
    confidence: float
    processing_time: float
    audio_duration: float
    word_count: int
    timestamp: str
    # 逐词置信度 (word, confidence)，来自 CTC 逐帧 softmax，用于在 UI 标注可疑片段
    word_confidences: List[Tuple[str, float]] = field(default_factory=list)


@dataclass
class OutpatientRecord:
    """门诊记录"""
    chief_complaint: str
    present_history: str
    past_history: str
    cardiovascular_exam: str
    ecg_findings: str
    assessment: str
    plan: str
    raw_response: str
    processing_time: float
    model_confidence: float
    template_used: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "chief_complaint": self.chief_complaint,
            "present_history": self.present_history,
            "past_history": self.past_history,
            "cardiovascular_exam": self.cardiovascular_exam,
            "ecg_findings": self.ecg_findings,
            "assessment": self.assessment,
            "plan": self.plan,
            "raw_response": self.raw_response,
            "processing_time": self.processing_time,
            "model_confidence": self.model_confidence,
            "template_used": self.template_used
        }
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


class BaseModelWrapper(ABC):
    """模型包装器基类"""
    
    def __init__(self, name: str, version: str, backend: InferenceBackend):
        self.name = name
        self.version = version
        self.backend = backend
        self.status = ModelStatus.UNINITIALIZED
        self.info = ModelInfo(
            name=name,
            version=version,
            backend=backend
        )
    
    @abstractmethod
    def load(self) -> bool:
        """加载模型"""
        pass
    
    @abstractmethod
    def unload(self) -> bool:
        """卸载模型"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """检查模型是否可用"""
        pass
    
    def _update_status(self, status: ModelStatus, error: Optional[str] = None):
        """更新状态"""
        self.status = status
        self.info.status = status
        if error:
            self.info.error_message = error
            logger.error(f"{self.name} error: {error}")


class MedASRWrapper(BaseModelWrapper):
    """
    MedASR 模型包装器
    使用 Transformers + PyTorch (MPS加速)
    """
    
    def __init__(self, model_id: str = "google/medasr"):
        super().__init__(
            name="MedASR",
            version="1.0.0",
            backend=InferenceBackend.TRANSFORMERS
        )
        self.model_id = model_id
        self.pipe = None
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.dtype = torch.float32
        # 转写前音频预处理（高通去嗡 + 峰值归一化）。可用环境变量关闭做 A/B
        self.preprocess_audio = os.environ.get("CARDIOVOICE_ASR_PREPROCESS", "1") != "0"

        logger.info(f"MedASR will use device: {self.device}")
    
    def load(self) -> bool:
        """加载 MedASR 模型"""
        start_time = time.time()
        self._update_status(ModelStatus.LOADING)
        
        try:
            logger.info(f"Loading MedASR model: {self.model_id}")
            
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model_id,
                device=self.device,
                torch_dtype=self.dtype,
                trust_remote_code=True
            )
            
            load_time = time.time() - start_time
            self.info.load_time = load_time
            self._update_status(ModelStatus.READY)
            
            logger.info(f"MedASR loaded successfully in {load_time:.2f}s")
            return True
            
        except Exception as e:
            self._update_status(ModelStatus.ERROR, str(e))
            logger.error(f"Failed to load MedASR: {e}")
            return False
    
    def unload(self) -> bool:
        """卸载 MedASR 模型"""
        try:
            if self.pipe:
                del self.pipe
                self.pipe = None

            # 释放加速器显存：本项目跑在 Apple Silicon (MPS) 上，原来误用了 cuda
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self._update_status(ModelStatus.UNLOADED)
            logger.info("MedASR unloaded")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unload MedASR: {e}")
            return False
    
    def is_available(self) -> bool:
        """检查 MedASR 是否可用"""
        return self.pipe is not None and self.status == ModelStatus.READY
    
    def transcribe_file(self, audio_path: str) -> TranscriptionResult:
        """
        转写音频文件
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            TranscriptionResult: 转写结果
        """
        if not self.is_available():
            raise RuntimeError("MedASR model not loaded")

        import soundfile as sf

        try:
            audio, sr = sf.read(audio_path, dtype="float32", always_2d=False)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != 16000:
                import soxr
                audio = soxr.resample(audio, sr, 16000)
            return self.transcribe_array(audio.astype(np.float32), sample_rate=16000)
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise

    # 特殊 token，不计入词
    _SPECIAL_TOKENS = {"<epsilon>", "<s>", "</s>", "<unk>", "<pad>"}

    def _decode_with_confidence(
        self, audio_float: np.ndarray, window_s: int = 20
    ) -> Tuple[str, List[Tuple[str, float]]]:
        """
        贪心 CTC 解码，返回 (文本, 逐词置信度)。

        置信度 = 每词所含 token 的逐帧 softmax 最大概率均值。CTC 空白符 (id 0)
        和特殊 token 被跳过；SentencePiece 的 '▁' 标记词边界。长音频按 window_s
        分窗解码后拼接。
        """
        model = self.pipe.model
        fe = self.pipe.feature_extractor
        tok = self.pipe.tokenizer

        win = int(16000 * window_s)
        words: List[Tuple[str, float]] = []

        for start in range(0, max(1, len(audio_float)), win):
            chunk = audio_float[start : start + win]
            if len(chunk) < 400:  # < 25ms，忽略
                continue
            inputs = fe(chunk, sampling_rate=16000, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = model(**inputs).logits
            probs = torch.softmax(logits.float(), dim=-1)[0]
            conf, ids = probs.max(dim=-1)
            ids = ids.cpu().numpy()
            conf = conf.cpu().numpy()

            prev = -1
            cur, cur_conf = "", []
            for i, c in zip(ids, conf):
                if i == prev:
                    continue
                prev = i
                if i == 0:  # blank
                    continue
                t = tok.convert_ids_to_tokens([int(i)])[0]
                if t in self._SPECIAL_TOKENS:
                    continue
                if t.startswith("▁"):
                    if cur:
                        words.append((cur, float(np.mean(cur_conf))))
                    cur, cur_conf = t[1:], [float(c)]
                else:
                    cur += t
                    cur_conf.append(float(c))
            if cur:
                words.append((cur, float(np.mean(cur_conf))))

        text = " ".join(w for w, _ in words)
        return text, words
    
    def transcribe_array(self, audio_data: np.ndarray, 
                         sample_rate: int = 16000) -> TranscriptionResult:
        """
        转写 numpy 音频数组
        
        Args:
            audio_data: 音频数据 (16-bit PCM)
            sample_rate: 采样率
            
        Returns:
            TranscriptionResult: 转写结果
        """
        if not self.is_available():
            raise RuntimeError("MedASR model not loaded")
        
        start_time = time.time()

        try:
            # 归一化到 [-1, 1]：整型按 16-bit PCM 缩放；浮点视为已归一化，避免重复缩放
            if np.issubdtype(audio_data.dtype, np.floating):
                audio_float = audio_data.astype(np.float32)
            else:
                audio_float = audio_data.astype(np.float32) / 32768.0

            if self.preprocess_audio:
                from preprocess import preprocess
                audio_float = preprocess(audio_float, sr=sample_rate)

            text, words = self._decode_with_confidence(audio_float)

            processing_time = time.time() - start_time

            # 真实置信度：逐词置信度均值（无词则为 0）
            confidence = (
                float(np.mean([c for _, c in words])) if words else 0.0
            )
            audio_duration = len(audio_data) / sample_rate

            transcription = TranscriptionResult(
                text=text,
                confidence=confidence,
                processing_time=processing_time,
                audio_duration=audio_duration,
                word_count=len(words),
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                word_confidences=words,
            )

            self.info.inference_count += 1
            self.info.last_inference_time = processing_time

            return transcription

        except Exception as e:
            logger.error(f"Array transcription failed: {e}")
            raise


class MedGemmaWrapper(BaseModelWrapper):
    """
    MedGemma 模型包装器
    使用 Ollama API + GGUF量化模型 (Metal加速)
    """
    
    def __init__(self,
                 model_name: Optional[str] = None,
                 temperature: float = 0.3,
                 max_tokens: int = 800):
        super().__init__(
            name="MedGemma",
            version="1.5.0",
            backend=InferenceBackend.OLLAMA
        )
        # 默认使用本地已有的 qwen3.5:9b；可通过环境变量 CARDIOVOICE_LLM_MODEL 覆盖
        if model_name is None:
            model_name = os.environ.get("CARDIOVOICE_LLM_MODEL", "qwen3.5:9b")
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.think = False  # 关闭推理模型(qwen3.5等)的思考输出，保证 content 非空
        self.ollama_model_name = None  # Ollama 中实际的模型名称
        
        # Ollama 服务地址
        self.ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    
    def load(self) -> bool:
        """检查并准备 MedGemma"""
        start_time = time.time()
        self._update_status(ModelStatus.LOADING)
        
        try:
            # 检查 Ollama 服务
            logger.info(f"Checking Ollama service at {self.ollama_host}")
            
            response = ollama.list()
            
            # Ollama 返回 ListResponse，模型名称在 .model 字段
            model_list = getattr(response, 'models', response)
            model_names = []
            
            for m in model_list:
                # 尝试多种可能的字段名
                model_name = getattr(m, 'model', None) or getattr(m, 'name', str(m))
                if model_name:
                    model_names.append(model_name)
            
            logger.info(f"Available models: {model_names}")
            
            # 规范化模型名称（去除 hf.co/ 前缀）
            def normalize_name(name):
                return name.replace('hf.co/', '').replace('hf.co:', '')
            
            # 检查目标模型是否存在
            target_name = self.model_name
            target_short = self.model_name.split(':')[0]
            
            # 检查ollama返回的每个模型
            for model in model_names:
                normalized = normalize_name(model)
                # 精确匹配或短名称匹配
                if (normalized == target_name or 
                    normalized == target_short or
                    model == target_name or 
                    model == target_short):
                    logger.info(f"Model {self.model_name} is available (found as {model})")
                    self.ollama_model_name = model  # 保存完整的 Ollama 模型名称
                    self._update_status(ModelStatus.READY)
                    self.info.load_time = time.time() - start_time
                    return True
            
            logger.warning(f"Model {target_name} not found in Ollama")
            logger.info(f"Available models: {model_names}")
            logger.info(f"Please pull the model: ollama pull {self.model_name}")
            self._update_status(ModelStatus.ERROR, "Model not found in Ollama")
            return False
                
        except Exception as e:
            self._update_status(ModelStatus.ERROR, str(e))
            logger.error(f"Failed to load MedGemma: {e}")
            return False
    
    def unload(self) -> bool:
        """Ollama 模型不需要显式卸载"""
        self._update_status(ModelStatus.UNLOADED)
        logger.info("MedGemma unloaded (Ollama manages memory)")
        return True
    
    def is_available(self) -> bool:
        """检查 MedGemma 是否可用"""
        try:
            if self.status != ModelStatus.READY:
                return False
            
            response = ollama.list()
            
            # Ollama 返回 ListResponse，模型名称在 .model 字段
            model_list = getattr(response, 'models', response)
            
            target_name = self.model_name
            target_short = self.model_name.split(':')[0]
            
            # 规范化模型名称
            def normalize_name(name):
                return name.replace('hf.co/', '').replace('hf.co:', '')
            
            for m in model_list:
                model_name = getattr(m, 'model', None) or getattr(m, 'name', str(m))
                if model_name:
                    normalized = normalize_name(model_name)
                    if (normalized == target_name or 
                        normalized == target_short or
                        model_name == target_name or 
                        model_name == target_short):
                        return True
            
            return False
            
        except Exception:
            return False
    
    def generate(self, prompt: str, 
                 options: Optional[Dict] = None) -> str:
        """
        生成文本
        
        Args:
            prompt: 提示词
            options: 生成选项 (可选)
            
        Returns:
            str: 生成的文本
        """
        if not self.is_available():
            raise RuntimeError("MedGemma model not loaded")
        
        # 合并选项
        gen_options = {
            'temperature': self.temperature,
            'num_predict': self.max_tokens,
        }
        if options:
            gen_options.update(options)
        
        try:
            response = ollama.chat(
                model=self.ollama_model_name,  # 使用完整的 Ollama 模型名称
                messages=[{
                    'role': 'user',
                    'content': prompt
                }],
                options=gen_options,
                think=self.think  # qwen3.5 等推理模型需关闭思考，否则 token 预算被思考耗尽，content 为空
            )

            self.info.inference_count += 1
            return response['message']['content']

        except Exception as e:
            logger.error(f"MedGemma generation failed: {e}")
            raise

    def correct_transcription(self, text: str,
                              terms: Optional[List[str]] = None) -> str:
        """
        用 LLM 保守地修正 ASR 转写中的明显错误（尤其医学术语/药名）。

        设计为"只改明显错、不改语义、不增删信息"。若返回结果异常（为空或长度
        相比原文剧烈缩短，疑似被概括/拒答），则回退为原文，保证绝不让结果更差。

        Args:
            text: 原始转写
            terms: 参考术语表（首选拼写）。为 None 时加载默认心内科词表。
        Returns:
            修正后的转写（失败时返回原文）
        """
        if not text or not text.strip():
            return text
        if not self.is_available():
            return text

        if terms is None:
            terms = load_lexicon()
        lexicon_str = ", ".join(terms) if terms else ""

        prompt = f"""You are a careful medical transcription editor. The text \
below is an automatic speech recognition (ASR) transcript of a cardiology \
consultation. It may contain misrecognized words, especially medical terms and \
drug names.

Fix ONLY clear ASR errors:
- Correct misheard/misspelled medical terms and drug names toward the reference \
list when clearly intended.
- Fix obvious word-level misrecognitions.

Do NOT:
- paraphrase, summarize, reorder, or change sentence structure
- add or remove any clinical information
- change numbers, doses, or units (unless an obvious ASR typo)
- expand or invent abbreviations
If a word is ambiguous, leave it unchanged.

Output ONLY the corrected transcript text — no preamble, no explanation.

Reference terms (preferred spellings): {lexicon_str}

Transcript:
{text}

Corrected transcript:"""

        try:
            corrected = self.generate(
                prompt, options={'temperature': 0.1}
            ).strip()
        except Exception as e:
            logger.error(f"Transcription correction failed: {e}")
            return text

        # 安全护栏：空、或长度相比原文缩水超过 40%（疑似被概括），一律回退原文
        if not corrected or len(corrected) < 0.6 * len(text):
            logger.warning("Correction output looked unsafe; keeping original")
            return text
        return corrected

    def generate_cardiology_record(self,
                                   transcription: str,
                                   patient_info: PatientInfo,
                                   template: str = "cardiology") -> OutpatientRecord:
        """
        生成心内科门诊记录
        
        Args:
            transcription: 语音转写文本
            patient_info: 患者信息
            template: 使用的模板
            
        Returns:
            OutpatientRecord: 生成的门诊记录
        """
        start_time = time.time()
        
        prompt = self._create_cardiology_prompt(transcription, patient_info)
        raw_response = self.generate(prompt)
        
        processing_time = time.time() - start_time
        self.info.last_inference_time = processing_time
        
        # 解析响应
        parsed = self._parse_response(raw_response)
        
        record = OutpatientRecord(
            chief_complaint=parsed.get('chief_complaint', ''),
            present_history=parsed.get('present_history', ''),
            past_history=parsed.get('past_history', ''),
            cardiovascular_exam=parsed.get('cardiovascular_exam', ''),
            ecg_findings=parsed.get('ecg_findings', ''),
            assessment=parsed.get('assessment', ''),
            plan=parsed.get('plan', ''),
            raw_response=raw_response,
            processing_time=processing_time,
            model_confidence=0.85,  # Ollama 不直接返回置信度
            template_used=template
        )
        
        logger.info(f"Cardiology record generated in {processing_time:.2f}s")
        return record
    
    def generate_general_record(self,
                                transcription: str,
                                patient_info: PatientInfo) -> OutpatientRecord:
        """生成通用门诊记录"""
        start_time = time.time()
        
        prompt = self._create_general_prompt(transcription, patient_info)
        raw_response = self.generate(prompt)
        
        processing_time = time.time() - start_time
        parsed = self._parse_response(raw_response)
        
        return OutpatientRecord(
            chief_complaint=parsed.get('chief_complaint', ''),
            present_history=parsed.get('present_history', ''),
            past_history=parsed.get('past_history', ''),
            cardiovascular_exam=parsed.get('cardiovascular_exam', ''),
            ecg_findings=parsed.get('ecg_findings', ''),
            assessment=parsed.get('assessment', ''),
            plan=parsed.get('plan', ''),
            raw_response=raw_response,
            processing_time=processing_time,
            model_confidence=0.85,
            template_used="general"
        )
    
    def _create_cardiology_prompt(self, transcription: str, 
                                  patient_info: PatientInfo) -> str:
        """创建心内科门诊记录提示词"""
        return f"""You are a board-certified cardiologist with 20 years of 
academic cardiology practice. Based on the following patient encounter 
transcription, generate a structured outpatient note.

REQUIREMENTS:
1. Structure: Chief Complaint, HPI, PMH, CV Exam, ECG, Assessment, Plan
2. Use precise cardiology terminology
3. Include relevant differential diagnoses
4. Be concise but comprehensive (200-400 words)

TRANSCRIPTION:
{transcription}

PATIENT INFORMATION:
- Name: {patient_info.name}
- Age: {patient_info.age}
- Gender: {patient_info.gender}
- Medical History: {patient_info.medical_history or 'Not provided'}
- Current Medications: {patient_info.medications or 'Not provided'}
- Allergies: {patient_info.allergies or 'None known'}

OUTPUT FORMAT:
CHIEF COMPLAINT:
[Main symptom in patient's own words]

HISTORY OF PRESENT ILLNESS:
[Detailed narrative with OLDCART format]

PAST MEDICAL HISTORY:
[Relevant conditions, surgeries]

CARDIOVASCULAR EXAMINATION:
[Heart sounds, murmurs, rhythm, pulses]

ECG FINDINGS:
[Rate, rhythm, axis, intervals, ST-T changes]

ASSESSMENT:
[Problem list with working diagnoses]

PLAN:
[Diagnostic tests, medications, follow-up]

Please generate the outpatient note:"""
    
    def _create_general_prompt(self, transcription: str,
                               patient_info: PatientInfo) -> str:
        """创建通用门诊记录提示词"""
        # 清理转写文本中的格式占位符
        clean_transcription = transcription
        placeholders = ['{period}', '{comma}', '{colon}', '{new paragraph}', 
                        '{open_bracket}', '{close_bracket}', '</s>']
        for ph in placeholders:
            clean_transcription = clean_transcription.replace(ph, '')
        
        return f"""You are an experienced physician. Based on the following 
medical report transcription, generate a concise structured outpatient note.

IMPORTANT: Output ONLY the final structured note. Do NOT include any instructions, 
reasoning, or analysis steps. Just output the note itself.

TRANSCRIPTION:
{clean_transcription}

PATIENT:
- Name: {patient_info.name}
- Age: {patient_info.age}
- Gender: {patient_info.gender}

OUTPUT (clean note only, no analysis):

CHIEF COMPLAINT:
[One sentence summary]

HISTORY OF PRESENT ILLNESS:
[2-3 sentences]

PAST MEDICAL HISTORY:
[Relevant conditions]

PHYSICAL EXAMINATION:
[Key findings]

ASSESSMENT:
[Diagnosis]

PLAN:
[Treatment and follow-up]"""
    
    def _parse_response(self, response: str) -> Dict[str, str]:
        """解析生成的响应"""
        import re
        
        # 清理响应中的格式占位符和常见问题
        placeholders = [
            '{period}', '{comma}', '{colon}', '{new paragraph}',
            '{open_bracket}', '{close_bracket}', '</s>',
            'The input is a', '2. **Analyze the Input', '3. **Address',
            'IMPORTANT:', 'OUTPUT (clean note only', 'OUTPUT:',
            'OUTPUT FORMAT:', 'Please generate', 'instructions:',
            'Think step by step', 'Let me analyze', 'Here is'
        ]
        cleaned_response = response
        for ph in placeholders:
            cleaned_response = cleaned_response.replace(ph, '')
        
        # 移除重复的空行
        cleaned_response = re.sub(r'\n{3,}', '\n\n', cleaned_response)
        cleaned_response = cleaned_response.strip()
        
        sections = {
            'chief_complaint': r'(?:CHIEF COMPLAINT|CC)[:\s]*(.*?)(?=\n[A-Z][A-Z\s]+:|\n\n|$)',
            'present_history': r'(?:HISTORY OF PRESENT ILLNESS|HPI)[:\s]*(.*?)(?=\n[A-Z][A-Z\s]+:|\n\n|$)',
            'past_history': r'(?:PAST MEDICAL HISTORY|PMH|Past History)[:\s]*(.*?)(?=\n[A-Z][A-Z\s]+:|\n\n|$)',
            'cardiovascular_exam': r'(?:CARDIOVASCULAR EXAMINATION|CV EXAM|Cardiovascular Exam|PHYSICAL EXAMINATION|EXAMINATION)[:\s]*(.*?)(?=\n[A-Z][A-Z\s]+:|\n\n|$)',
            'ecg_findings': r'(?:ECG FINDINGS|ECG)[:\s]*(.*?)(?=\n[A-Z][A-Z\s]+:|\n\n|$)',
            'assessment': r'(?:ASSESSMENT|DIAGNOSIS|IMPRESSION)[:\s]*(.*?)(?=\n[A-Z][A-Z\s]+:|\n\n|$)',
            'plan': r'(?:PLAN|TREATMENT|FOLLOW-UP)[:\s]*(.*?)(?=\n[A-Z][A-Z\s]+:|\n\n|$)',
        }
        
        parsed = {}
        for section_name, pattern in sections.items():
            match = re.search(pattern, cleaned_response, re.DOTALL | re.IGNORECASE)
            if match:
                content = match.group(1).strip()
                # 再次清理内容中的占位符
                for ph in placeholders:
                    content = content.replace(ph, '')
                # 移除过多的空行
                content = re.sub(r'\n{3,}', '\n\n', content)
                # 去掉开头残留的 markdown 标记 (** / # / :)
                content = re.sub(r'^[\s*#:]+', '', content).strip()
                parsed[section_name] = content
            else:
                parsed[section_name] = ""
        
        return parsed


class UnifiedModelManager:
    """
    统一模型管理器
    
    协调 MedASR 和 MedGemma 的加载、调用和状态管理
    """
    
    def __init__(self):
        self.medasr = MedASRWrapper()
        self.medgemma = MedGemmaWrapper()
        
        self._lock = threading.Lock()
        self._callbacks: Dict[str, List[Callable]] = {}
    
    def load_all(self, verbose: bool = True) -> Dict[str, bool]:
        """
        加载所有模型
        
        Args:
            verbose: 是否打印详细信息
            
        Returns:
            Dict: 各模型的加载状态
        """
        results = {}

        # 并行加载：MedASR 权重加载较慢(~15s)，MedGemma 只是探测 Ollama，
        # 两者无依赖，并行可缩短启动时间。
        if verbose:
            logger.info("Loading MedASR (Transformers) + MedGemma (Ollama) in parallel...")

        def _load_medasr():
            results['medasr'] = self.medasr.load()

        def _load_medgemma():
            results['medgemma'] = self.medgemma.load()

        threads = [
            threading.Thread(target=_load_medasr),
            threading.Thread(target=_load_medgemma),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 检查整体状态
        all_ready = all(results.values())
        if verbose:
            if all_ready:
                logger.info("✓ All models loaded successfully")
            else:
                failed = [k for k, v in results.items() if not v]
                logger.warning(f"✗ Some models failed to load: {failed}")
        
        return results
    
    def unload_all(self) -> None:
        """卸载所有模型"""
        with self._lock:
            self.medasr.unload()
            self.medgemma.unload()
            logger.info("All models unloaded")
    
    def is_ready(self) -> bool:
        """检查是否所有模型都已就绪"""
        return self.medasr.is_available() and self.medgemma.is_available()
    
    def get_status(self) -> Dict[str, ModelInfo]:
        """获取所有模型状态"""
        return {
            'medasr': self.medasr.info,
            'medgemma': self.medgemma.info
        }
    
    def transcribe(self, audio_input: Union[str, np.ndarray]) -> TranscriptionResult:
        """
        语音转文字
        
        Args:
            audio_input: 音频文件路径或 numpy 数组
            
        Returns:
            TranscriptionResult: 转写结果
        """
        if isinstance(audio_input, str):
            return self.medasr.transcribe_file(audio_input)
        else:
            return self.medasr.transcribe_array(audio_input)
    
    def generate_record(self,
                        transcription: str,
                        patient_info: PatientInfo,
                        template: str = "cardiology") -> OutpatientRecord:
        """
        生成门诊记录
        
        Args:
            transcription: 语音转写文本
            patient_info: 患者信息
            template: 使用的模板 ('cardiology' or 'general')
            
        Returns:
            OutpatientRecord: 生成的门诊记录
        """
        if template == "cardiology":
            return self.medgemma.generate_cardiology_record(
                transcription, patient_info, template
            )
        else:
            return self.medgemma.generate_general_record(
                transcription, patient_info
            )
    
    def full_pipeline(self,
                      audio_input: Union[str, np.ndarray],
                      patient_info: PatientInfo,
                      template: str = "cardiology") -> Dict[str, Any]:
        """
        完整流程：转写 + 生成门诊记录
        
        Args:
            audio_input: 音频文件或数组
            patient_info: 患者信息
            template: 门诊记录模板
            
        Returns:
            Dict: 包含转写结果和门诊记录的字典
        """
        start_time = time.time()
        
        # Step 1: 转写
        transcription = self.transcribe(audio_input)
        
        # Step 2: 生成门诊记录
        record = self.generate_record(
            transcription.text,
            patient_info,
            template
        )
        
        total_time = time.time() - start_time
        
        return {
            'transcription': transcription,
            'record': record,
            'total_time': total_time,
            'patient_info': patient_info
        }
    
    def register_callback(self, event: str, callback: Callable) -> None:
        """注册事件回调"""
        if event not in self._callbacks:
            self._callbacks[event] = []
        self._callbacks[event].append(callback)
    
    def _trigger_callbacks(self, event: str, *args, **kwargs) -> None:
        """触发事件回调"""
        for callback in self._callbacks.get(event, []):
            try:
                callback(*args, **kwargs)
            except Exception as e:
                logger.error(f"Callback error in {event}: {e}")


_LEXICON_CACHE: Optional[List[str]] = None


def load_lexicon(path: Optional[str] = None) -> List[str]:
    """加载术语表（默认 lexicons/cardiology.txt）。忽略空行与 # 注释。"""
    global _LEXICON_CACHE
    if path is None and _LEXICON_CACHE is not None:
        return _LEXICON_CACHE
    if path is None:
        path = str(Path(__file__).resolve().parent.parent / "lexicons" / "cardiology.txt")
    terms: List[str] = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    terms.append(line)
    except FileNotFoundError:
        logger.warning(f"Lexicon not found: {path}")
    if _LEXICON_CACHE is None:
        _LEXICON_CACHE = terms
    return terms


def create_patient_info(name: str, age: int, gender: str,
                       medical_history: str = None,
                       medications: str = None,
                       allergies: str = None) -> PatientInfo:
    """创建患者信息对象的便捷函数"""
    return PatientInfo(
        name=name,
        age=age,
        gender=gender,
        medical_history=medical_history,
        medications=medications,
        allergies=allergies
    )


# ============================================================
# 使用示例
# ============================================================

if __name__ == "__main__":
    # 这是一个使用示例，展示如何调用统一管理器
    # 实际使用时，请取消注释并运行
    
    print("""
    CardioVoice Unified Model Manager - 使用示例
    =============================================
    
    # 1. 创建管理器
    manager = UnifiedModelManager()
    
    # 2. 加载所有模型
    results = manager.load_all()
    print(results)  # {'medasr': True, 'medgemma': True}
    
    # 3. 检查状态
    status = manager.get_status()
    print(status)
    
    # 4. 转写音频
    # transcription = manager.transcribe("path/to/audio.wav")
    # print(transcription.text)
    
    # 5. 生成门诊记录
    # patient = create_patient_info(
    #     name="张三",
    #     age=58,
    #     gender="male",
    #     medical_history="高血压5年",
    #     medications="苯磺酸氨氯地平片"
    # )
    # 
    # record = manager.generate_record(
    #     transcription="医生：您好，哪里不舒服？患者：胸痛3天了...",
    #     patient_info=patient,
    #     template="cardiology"
    # )
    # 
    # print(record.to_json())
    
    # 6. 完整流程
    # result = manager.full_pipeline(
    #     audio_input="path/to/audio.wav",
    #     patient_info=patient,
    #     template="cardiology"
    # )
    
    # 7. 卸载模型
    # manager.unload_all()
    
    注意：实际运行时，请确保：
    1. Ollama 服务正在运行 (ollama serve)
    2. 已拉取模型 (ollama pull unsloth/medgemma-1.5-4b-it-GGUF:Q4_K_M)
    3. 已安装依赖 (pip install torch transformers ollama)
    """)
