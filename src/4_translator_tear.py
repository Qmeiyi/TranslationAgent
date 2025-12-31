#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TEaR翻译脚本 (Agent 2: Translation & Refinement Agent)
功能：实现翻译-评估-润色的循环流程，包含回译验证
"""

import json
import os
import time
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# --- 📁 配置路径 ---
INPUT_FILE = "../data/processed/诡秘之主_final.jsonl"
GLOSSARY_FILE = "../data/glossary/project_knowledge_base.json"
OUTPUT_FILE = "../data/output/诡秘之主_tear_result.jsonl"

# --- 🤖 初始化模型 (DeepSeek) ---
def init_llm(model_name: str = "deepseek-chat", temperature: float = 0.3):
    """
    初始化LLM模型
    
    Args:
        model_name: 模型名称
        temperature: 温度参数
        
    Returns:
        初始化后的LLM实例
    """
    return ChatOpenAI(
        model="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
        api_key="sk-cautwxmuhdpxhtuilctlfpecaoxpzhagpzfzmkdxgrywjpum", 
        base_url="https://api.siliconflow.cn/v1/",
        temperature=0.1 # 翻译任务需要低温度，保持严谨
    )

# --- 📖 辅助函数：加载术语表 ---
def load_glossary_context():
    """
    加载术语表上下文
    
    Returns:
        格式化的术语表文本
    """
    if not os.path.exists(GLOSSARY_FILE):
        return "No glossary available."
    
    with open(GLOSSARY_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 格式化为 Markdown 列表，强化 Prompt 的注意力
    lines = ["## 强制术语表 (Strict Glossary):"]
    for term in data.get("terms", []):
        lines.append(f"- **{term['term']}**: {term['suggested_translation']} ({term['category']})")
    
    # 加入世界观摘要，帮助模型定调
    world_info = data.get("world_summary", "")
    return "\n".join(lines) + f"\n\n## 世界观背景:\n{world_info}"

GLOSSARY_CONTEXT = load_glossary_context()

# --- 🔄 回译验证函数 ---
def back_translate(text: str, llm) -> str:
    """
    将文本回译，用于验证翻译质量
    
    Args:
        text: 要回译的文本
        llm: LLM实例
        
    Returns:
        回译后的文本
    """
    back_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一位专业的翻译专家。请将下面的英文文本准确地翻译回中文。"),
        ("user", "{text}")
    ])
    
    back_chain = back_prompt | llm | StrOutputParser()
    return back_chain.invoke({"text": text})

# ==============================================================================
# 🎭 步骤 1: 初稿生成 (Drafting)
# ==============================================================================
draft_prompt = ChatPromptTemplate.from_messages([
    ("system", """
    你是一位精通中英翻译的小说家，擅长“维多利亚奇幻”与“克苏鲁神话”风格。
    你的任务是将中文小说《诡秘之主》翻译成英文。
    
    ### 风格要求 (Style Guide):
    1. **氛围**: 保持神秘、压抑、复古的英伦风 (Victorian Era)。
    2. **用词**: 使用狄更斯或柯南·道尔式的词汇（例如用 'Crimson' 而非 'Red'，用 'Revolver' 而非 'Gun'）。
    3. **忠实**: 保留原文的叙事节奏和伏笔。
    
    {glossary}
    """),
    ("user", "【章节标题】: {title}\n\n【原文内容】:\n{text}\n\n请直接输出英文初稿，不要包含任何解释。")
])

# ==============================================================================
# 🧐 步骤 2: 审校与反思 (Critique / Self-Reflection)
# ==============================================================================
critique_prompt = ChatPromptTemplate.from_messages([
    ("system", """
    你是一位严苛的翻译审校专家。你的任务是检查一份《诡秘之主》的英文草稿。
    
    请重点检查以下问题：
    1. **术语一致性**: 是否严格遵守了以下术语表？(例如: '值夜者' 必须是 'Nighthawks')。
    2. **漏译/错译**: 是否有遗漏的段落或明显的语义错误？
    3. **风格违和**: 是否出现了过于现代的美式俚语（如 'Okay', 'Cool'）？
    4. **回译验证**: 将英文译文反向翻译回中文，与原文对比，检查语义偏差。
    
    {glossary}
    """),
    ("user", "【原文】:\n{original}\n\n【英文初稿】:\n{draft}\n\n【回译中文】:\n{back_translation}\n\n请列出具体的修改建议（如果翻译完美，请直接回复 'PASS'）。")
])

# ==============================================================================
# ✨ 步骤 3: 最终润色 (Refinement)
# ==============================================================================
refine_prompt = ChatPromptTemplate.from_messages([
    ("system", """
    你是一位追求完美的文学编辑。你需要根据审校意见，重写并润色译文。
    
    {glossary}
    """),
    ("user", "【原文】:\n{original}\n\n【初稿】:\n{draft}\n\n【审校意见】:\n{critique}\n\n请输出最终版本的英文译文。")
])

# --- ⛓️ 构建 Chain ---
llm_draft = init_llm(temperature=0.3)
llm_critique = init_llm(temperature=0.1)
llm_refine = init_llm(temperature=0.2)
llm_backtranslate = init_llm(temperature=0.0)

draft_chain = draft_prompt | llm_draft | StrOutputParser()
critique_chain = critique_prompt | llm_critique | StrOutputParser()
refine_chain = refine_prompt | llm_refine | StrOutputParser()

# ==============================================================================
# 🚀 主程序：执行 TEaR 循环
# ==============================================================================
def process_translation():
    """
    执行TEaR翻译循环
    """
    print(f"📚 载入术语表，包含 {GLOSSARY_CONTEXT.count('- **')} 个核心词条。")
    print("🚀 启动 TEaR (Translate-Evaluate-Refine) 引擎...\n")

    # 创建输出目录
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    with open(INPUT_FILE, 'r', encoding='utf-8') as fin, \
         open(OUTPUT_FILE, 'w', encoding='utf-8') as fout:
        
        for line in fin:
            chapter = json.loads(line)
            title = chapter['title']
            text = chapter['text']
            
            print(f"⏳ [1/4] 正在生成初稿: {title} ...")
            
            # Step 1: Draft
            draft = draft_chain.invoke({
                "title": title, 
                "text": text, 
                "glossary": GLOSSARY_CONTEXT
            })
            
            print(f"🔄 [2/4] 正在执行回译验证...")
            # Step 2: Back Translation
            back_translation = back_translate(draft, llm_backtranslate)
            
            print(f"🧐 [3/4] 正在自我审校...")
            # Step 3: Critique
            critique = critique_chain.invoke({
                "original": text, 
                "draft": draft, 
                "back_translation": back_translation,
                "glossary": GLOSSARY_CONTEXT
            })
            
            final_translation = draft
            
            # 只有当审校发现问题时，才执行 Refine (节省 Token)
            if "PASS" not in critique and len(critique) > 10:
                print(f"🔧 [4/4] 发现改进点，正在润色...")
                # Step 4: Refine
                final_translation = refine_chain.invoke({
                    "original": text,
                    "draft": draft,
                    "critique": critique,
                    "glossary": GLOSSARY_CONTEXT
                })
            else:
                print(f"✅ [4/4] 初稿质量完美，跳过润色。")

            # 保存结果（包含中间过程，方便写报告对比）
            result = {
                "chapter_index": chapter['chapter_index'],
                "title": title,
                "draft": draft,
                "back_translation": back_translation,
                "critique": critique,
                "final_translation": final_translation
            }
            
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            fout.flush()  # 实时保存
            
            print("-" * 50)

if __name__ == "__main__":
    process_translation()