#!/usr/bin/env python3
"""
CardioVoice Demo and Test Script
=================================

使用真实测试音频测试统一模型管理器 (UnifiedModelManager)。

Usage:
    python demo_unified.py --test real     # 使用真实音频测试
    python demo_unified.py --mode demo     # 代码演示（默认）
    python demo_unified.py --mode full     # 完整演示

Author: CardioVoice Team
Date: January 2026
Version: 2.0.0
"""

import argparse
import sys
import time
import json
from pathlib import Path

# 添加 backend 目录到路径
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from unified_model_manager import (
    UnifiedModelManager,
    PatientInfo,
    create_patient_info,
    ModelStatus,
    InferenceBackend
)

# ============================================================================
# 测试音频路径 - 使用 MedASR 官方测试音频
# ============================================================================
TEST_AUDIO_PATH = (
    "/Users/lizhanbing12/.cache/huggingface/hub/models--google--medasr"
    "/snapshots/2625be4f1377ac544b451c6938eaf955c19a9c38/test_audio.wav"
)


def check_test_audio():
    """检查测试音频文件是否存在"""
    import os
    if os.path.exists(TEST_AUDIO_PATH):
        print(f"✓ 测试音频文件存在: {TEST_AUDIO_PATH}")
        return True
    else:
        print(f"✗ 测试音频文件不存在: {TEST_AUDIO_PATH}")
        return False


def run_real_test():
    """
    使用真实测试音频运行完整测试流程
    """
    print("\n" + "=" * 70)
    print("  🏥 CardioVoice - 真实测试模式")
    print("  使用 MedASR 官方测试音频")
    print("=" * 70)
    
    # 检查测试音频
    if not check_test_audio():
        print("\n请检查 MedASR 模型是否正确下载")
        return False
    
    # 创建管理器
    manager = UnifiedModelManager()
    
    print("\n" + "-" * 50)
    print("步骤 1: 加载模型")
    print("-" * 50)
    
    results = manager.load_all(verbose=True)
    
    if not all(results.values()):
        print("\n✗ 模型加载失败")
        status = manager.get_status()
        for name, info in status.items():
            if info.status != ModelStatus.READY:
                print(f"  {name}: {info.status.value} - {info.error_message}")
        return False
    
    print("\n✓ 所有模型加载成功!")
    
    print("\n" + "-" * 50)
    print("步骤 2: MedASR 语音转写测试")
    print("-" * 50)
    
    try:
        print(f"\n正在转写音频: {TEST_AUDIO_PATH}")
        start_time = time.time()
        transcription = manager.transcribe(TEST_AUDIO_PATH)
        transcribe_time = time.time() - start_time
        
        print(f"\n✓ 转写完成! 耗时: {transcribe_time:.2f}秒")
        print(f"\n转写结果:")
        print("-" * 50)
        print(transcription.text)
        print("-" * 50)
        print(f"\n统计信息:")
        print(f"  - 置信度: {transcription.confidence:.2%}")
        print(f"  - 处理时间: {transcription.processing_time:.2f}秒")
        print(f"  - 音频时长: {transcription.audio_duration:.1f}秒")
        print(f"  - 字数: {transcription.word_count}")
        
    except Exception as e:
        print(f"\n✗ 转写失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "-" * 50)
    print("步骤 3: MedGemma 门诊记录生成测试")
    print("-" * 50)
    
    # 创建测试患者信息
    patient = create_patient_info(
        name="测试患者",
        age=58,
        gender="male",
        medical_history="高血压5年，规律服药",
        medications="苯磺酸氨氯地平片 10mg qd",
        allergies="无"
    )
    
    try:
        print("\n正在生成门诊记录...")
        start_time = time.time()
        
        # 使用通用模板生成
        record = manager.generate_record(
            transcription=transcription.text,
            patient_info=patient,
            template="general"
        )
        
        gen_time = time.time() - start_time
        
        print(f"\n✓ 门诊记录生成完成! 耗时: {gen_time:.2f}秒")
        print(f"\n生成的门诊记录:")
        print("=" * 50)
        
        fields = [
            ("主诉", record.chief_complaint),
            ("现病史", record.present_history),
            ("既往史", record.past_history),
            ("检查", record.cardiovascular_exam),
            ("诊断", record.assessment),
            ("处理计划", record.plan),
        ]
        
        for field_name, field_value in fields:
            if field_value:
                print(f"\n{field_name}:")
                print(f"  {field_value}")
        
        print("=" * 50)
        print(f"\n统计信息:")
        print(f"  - 处理时间: {record.processing_time:.2f}秒")
        print(f"  - 使用模板: {record.template_used}")
        print(f"  - 原始响应长度: {len(record.raw_response)} 字符")
        
    except Exception as e:
        print(f"\n✗ 门诊记录生成失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "-" * 50)
    print("步骤 4: 保存测试结果")
    print("-" * 50)
    
    # 保存结果到文件
    output_data = {
        "test_info": {
            "test_audio": TEST_AUDIO_PATH,
            "test_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "transcription": {
            "text": transcription.text,
            "confidence": transcription.confidence,
            "processing_time": transcription.processing_time,
            "audio_duration": transcription.audio_duration,
            "word_count": transcription.word_count,
        },
        "patient": {
            "name": patient.name,
            "age": patient.age,
            "gender": patient.gender,
            "medical_history": patient.medical_history,
            "medications": patient.medications,
        },
        "record": record.to_dict(),
        "timing": {
            "transcription_seconds": transcribe_time,
            "generation_seconds": gen_time,
            "total_seconds": transcribe_time + gen_time,
        }
    }
    
    output_file = "test_result.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ 测试结果已保存到: {output_file}")
    
    print("\n" + "=" * 70)
    print("  ✅ 测试完成!")
    print("=" * 70)
    print(f"\n总耗时: {transcribe_time + gen_time:.2f}秒")
    print(f"  - 转写: {transcribe_time:.2f}秒")
    print(f"  - 生成: {gen_time:.2f}秒")
    
    return True


def print_header(title: str, width: int = 70):
    """打印格式化的标题"""
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_subheader(title: str):
    """打印格式化的子标题"""
    print(f"\n--- {title} ---")


def print_code_block(code: str, language: str = "python"):
    """打印代码块"""
    print(f"\n```{language}")
    print(code)
    print("```")


def demo_manager_initialization():
    """演示管理器初始化"""
    print_header("1. UnifiedModelManager 初始化")
    
    print("""
    UnifiedModelManager 是一个统一的接口，用于管理 MedASR 和 MedGemma 两个模型。
    它协调两个后端的加载、调用和状态管理。
    """)
    
    code = '''
    from unified_model_manager import UnifiedModelManager

    # 创建管理器实例
    manager = UnifiedModelManager()

    # 管理器内部结构
    # manager.medasr    → MedASRWrapper (Transformers)
    # manager.medgemma  → MedGemmaWrapper (Ollama)
    '''
    print_code_block(code)


def demo_model_loading():
    """演示模型加载"""
    print_header("2. 模型加载")
    
    print("""
    load_all() 方法会依次加载两个模型，并返回加载状态。
    加载过程会检查：
    - MedASR: Transformers 库和模型权重
    - MedGemma: Ollama 服务和模型是否存在
    """)
    
    code = '''
    # 创建管理器
    manager = UnifiedModelManager()

    # 加载所有模型
    results = manager.load_all(verbose=True)

    # 检查整体状态
    if manager.is_ready():
        print("✓ 所有模型已就绪")
    '''
    print_code_block(code)


def demo_medasr_usage():
    """演示 MedASR 使用"""
    print_header("3. MedASR 语音转文字")
    
    print("""
    MedASRWrapper 提供了两种转写方式：
    1. transcribe_file(): 从音频文件转写
    2. transcribe_array(): 从 numpy 数组转写
    """)
    
    code = '''
    from unified_model_manager import UnifiedModelManager

    manager = UnifiedModelManager()
    manager.load_all()

    # 从音频文件转写
    result = manager.transcribe("path/to/patient_audio.wav")

    print(f"转写文本: {result.text}")
    print(f"置信度: {result.confidence:.2%}")
    '''
    print_code_block(code)


def demo_medgemma_usage():
    """演示 MedGemma 使用"""
    print_header("4. MedGemma 门诊记录生成")
    
    code = '''
    from unified_model_manager import UnifiedModelManager, create_patient_info

    manager = UnifiedModelManager()
    manager.load_all()

    patient = create_patient_info(
        name="张三",
        age=58,
        gender="male"
    )

    record = manager.generate_record(
        transcription="医生：您好...",
        patient_info=patient,
        template="cardiology"
    )

    print(f"主诉: {record.chief_complaint}")
    print(f"诊断: {record.assessment}")
    '''
    print_code_block(code)


def demo_full_pipeline():
    """演示完整流程"""
    print_header("5. 完整流程演示")
    
    code = '''
    from unified_model_manager import UnifiedModelManager, create_patient_info

    manager = UnifiedModelManager()
    manager.load_all()

    patient = create_patient_info(name="李四", age=65, gender="female")

    result = manager.full_pipeline(
        audio_input="path/to/audio.wav",
        patient_info=patient,
        template="general"
    )

    print(f"转写: {result['transcription'].text}")
    print(f"门诊记录: {result['record'].chief_complaint}")
    '''
    print_code_block(code)


def demo_model_status():
    """演示模型状态查询"""
    print_header("6. 模型状态监控")
    
    code = '''
    from unified_model_manager import UnifiedModelManager

    manager = UnifiedModelManager()
    manager.load_all()

    status = manager.get_status()
    for name, info in status.items():
        print(f"{name}: {info.status.value}")
        print(f"  推理次数: {info.inference_count}")
    '''
    print_code_block(code)


def main():
    parser = argparse.ArgumentParser(
        description="CardioVoice Demo and Test Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 使用真实测试音频运行完整测试
  python demo_unified.py --test real
  
  # 代码演示模式
  python demo_unified.py --mode demo       # 默认，仅显示代码
  python demo_unified.py --mode full       # 完整演示
  python demo_unified.py --mode medasr     # MedASR 演示
  python demo_unified.py --mode medgemma   # MedGemma 演示
        """
    )
    
    parser.add_argument(
        '--test',
        type=str,
        choices=['real'],
        help='Run actual test with real audio'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['demo', 'full', 'manager', 'medasr', 'medgemma', 'pipeline', 'status'],
        default='demo',
        help='Demo mode (default: demo)'
    )
    
    args = parser.parse_args()
    
    # 如果是真实测试模式
    if args.test == 'real':
        success = run_real_test()
        sys.exit(0 if success else 1)
    
    print("\n" + "🏥" * 25)
    print("\n  CardioVoice Demo - 统一模型管理器演示")
    print("  MedASR + MedGemma 集成方案\n")
    print("  " + "-" * 50)
    
    if args.mode == 'full' or args.mode == 'manager':
        demo_manager_initialization()
        demo_model_loading()
    
    if args.mode == 'full' or args.mode == 'medasr':
        demo_medasr_usage()
    
    if args.mode == 'full' or args.mode == 'medgemma':
        demo_medgemma_usage()
    
    if args.mode == 'full' or args.mode == 'pipeline':
        demo_full_pipeline()
    
    if args.mode == 'full' or args.mode == 'status':
        demo_model_status()
    
    print_header("Demo 完成")
    print("""
    运行真实测试:
    1. 确保 Ollama 正在运行: ollama serve
    2. 激活环境: conda activate medgemma
    3. 安装依赖: pip install ollama
    4. 运行: python demo_unified.py --test real
    
    测试音频: {test_audio}
    
    完整文档请参考:
    - README.md
    - backend/unified_model_manager.py (源代码)
    """.format(test_audio=TEST_AUDIO_PATH))


if __name__ == "__main__":
    main()
