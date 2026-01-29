#!/usr/bin/env python3
"""
MedGemma 1.5 测试脚本
测试医学文本生成功能
"""

from transformers import pipeline
import torch

def test_medgemma():
    """测试MedGemma 1.5模型"""
    print("=" * 50)
    print("MedGemma 1.5 测试")
    print("=" * 50)

    # 检查设备
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\n📱 使用设备: {device.upper()}")

    # 加载模型
    print("\n⏳ 正在加载MedGemma 1.5-4B-IT模型...")
    print("⚠️  注意: 首次运行会下载约8GB的模型文件")

    try:
        pipe = pipeline(
            "text-generation",
            model="google/medgemma-1.5-4b-it",
            torch_dtype=torch.bfloat16,
            device=device,
        )

        print("✅ 模型加载成功！")

        # 测试医学问答
        print("\n" + "=" * 50)
        print("测试1: 医学知识问答")
        print("=" * 50)

        question1 = "如何区分细菌性肺炎和病毒性肺炎？请用简洁的语言回答。"
        print(f"\n❓ 问题: {question1}")

        messages1 = [
            {
                "role": "user",
                "content": question1
            }
        ]

        output1 = pipe(text=messages1, max_new_tokens=300)
        answer1 = output1[0]["generated_text"][-1]["content"]
        print(f"\n💡 回答:\n{answer1}")

        # 测试门诊记录生成
        print("\n" + "=" * 50)
        print("测试2: 门诊记录生成")
        print("=" * 50)

        patient_info = """
        患者信息：
        - 姓名：张三
        - 年龄：45岁
        - 性别：男
        - 主诉：发热、咳嗽3天
        - 现病史：患者3天前无明显诱因出现发热，最高体温38.5℃，伴有咳嗽，咳少量白痰，无胸闷、气促。
        - 既往史：高血压病史5年，规律服药控制。
        """

        prompt2 = f"""
        你是一名专业的门诊医生。根据以下患者信息，生成规范的门诊记录。
        要求：
        1. 结构化输出，包括：主诉、现病史、既往史、初步诊断、处理意见
        2. 语言专业、准确
        3. 简洁明了

        {patient_info}
        """

        print(f"\n📋 患者信息已提供")

        messages2 = [
            {
                "role": "user",
                "content": prompt2
            }
        ]

        output2 = pipe(text=messages2, max_new_tokens=500)
        answer2 = output2[0]["generated_text"][-1]["content"]
        print(f"\n📄 生成的门诊记录:\n{answer2}")

        print("\n" + "=" * 50)
        print("✅ 所有测试通过！")
        print("=" * 50)

    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
        print("\n可能的解决方案:")
        print("1. 检查网络连接")
        print("2. 确认Hugging Face API密钥（如需要）")
        print("3. 检查磁盘空间（至少需要8GB）")
        return False

    return True

if __name__ == "__main__":
    success = test_medgemma()
    exit(0 if success else 1)
