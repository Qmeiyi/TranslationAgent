#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基线翻译脚本 (Baseline Translator)
功能：不挂载术语表，不使用 TEaR，直接用 gpt-3.5 或 deepseek-chat 进行单次翻译
用于生成“差生”结果，衬托 TEaR 翻译的“优等生”表现
"""

import json
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# --- 📁 配置路径 ---
INPUT_FILE = "../data/processed/诡秘之主_final.jsonl"
OUTPUT_FILE = "../data/output/诡秘之主_baseline_result.jsonl"

# --- 🤖 初始化模型 ---
def init_baseline_llm():
    """
    初始化基线翻译模型
    
    Returns:
        初始化后的LLM实例
    """
    return ChatOpenAI(
        model="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
        api_key="sk-cautwxmuhdpxhtuilctlfpecaoxpzhagpzfzmkdxgrywjpum", 
        base_url="https://api.siliconflow.cn/v1/",
        temperature=0.1 # 翻译任务需要低温度，保持严谨
    )

# --- 🎭 基线翻译提示词 ---
baseline_prompt = ChatPromptTemplate.from_messages([
    ("system", """
    你是一位中英翻译专家。请将下面的中文小说翻译成英文。
    
    风格要求：
    - 保持原文意思准确
    - 使用流畅的英文表达
    - 无需特别的风格要求
    """),
    ("user", "【章节标题】: {title}\n\n【原文内容】:\n{text}\n\n请直接输出英文翻译，不要包含任何解释。")
])

# --- ⛓️ 构建基线翻译 Chain ---
llm = init_baseline_llm()
baseline_chain = baseline_prompt | llm | StrOutputParser()

# --- 🚀 主程序 ---
def main():
    """
    主函数：执行基线翻译
    """
    print("📝 启动基线翻译 (无术语表，无 TEaR)...")
    
    # 创建输出目录
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    processed_count = 0
    with open(INPUT_FILE, 'r', encoding='utf-8') as fin, \
         open(OUTPUT_FILE, 'w', encoding='utf-8') as fout:
        
        for line in fin:
            chapter = json.loads(line)
            title = chapter['title']
            text = chapter['text']
            
            print(f"⏳ 正在翻译章节: {title} ...")
            
            # 直接调用模型进行单次翻译（无术语表，无TEaR循环）
            baseline_translation = baseline_chain.invoke({
                "title": title,
                "text": text
            })
            
            # 保存结果
            result = {
                "chapter_index": chapter['chapter_index'],
                "title": title,
                "baseline_translation": baseline_translation
            }
            
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            fout.flush()  # 实时保存
            
            processed_count += 1
            print(f"✅ 完成章节: {title}")
            print("-" * 50)
    
    print(f"📊 基线翻译完成！共处理 {processed_count} 章。")
    print(f"📁 结果已保存到: {OUTPUT_FILE}")
    print("💡 提示：此基线翻译用于与 TEaR 翻译结果进行对比分析")

if __name__ == "__main__":
    main()