#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
术语提取脚本 (Agent 1: Terminology & Entity Agent)
功能：识别并规范化小说中的术语、实体和文化负载词
"""

import json
import os
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser

# --- 1. 定义数据结构 ---
class TermEntry(BaseModel):
    """术语条目数据结构"""
    term: str = Field(description="原文术语")
    category: str = Field(description="类别: Person, Location, Org, Concept, Item, Currency")
    definition: str = Field(description="结合全书上下文的深度定义")
    suggested_translation: str = Field(description="建议的英文译名 (需保持全书一致)")
    context_clue: Optional[str] = Field(description="该术语首次出现或最关键的原文片段引用", default=None)

class KnowledgeGraph(BaseModel):
    """知识库数据结构"""
    world_summary: str = Field(description="对前10章世界观、力量体系的简要总结（200字以内）")
    terms: List[TermEntry]

# --- 2. 初始化模型 --- 
def init_llm():
    """初始化LLM模型"""
    return ChatOpenAI(
        model="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
        api_key="sk-cautwxmuhdpxhtuilctlfpecaoxpzhagpzfzmkdxgrywjpum", 
        base_url="https://api.siliconflow.cn/v1/",
        temperature=0.1,
        max_tokens=8000
    )

# --- 3. 加载文本 --- 
def load_full_text(filepath: str) -> str:
    """
    加载完整文本用于术语提取
    
    Args:
        filepath: 文本文件路径
        
    Returns:
        拼接好的完整文本
    """
    full_text = ""
    print("📚 正在加载全量文本到内存...")
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            # 拼接格式： [第X章 标题] 
            full_text += f"\n\n[第{data['chapter_index']}章 {data['title']}]\n{data['text']}"
            full_text += f"\n\n[第{data['chapter_index']}章 {data['title']}]\n{data['text']}"
    
    token_est = len(full_text) 
    print(f"✅ 加载完成！总字符数: {len(full_text)} (预估 Token: {token_est // 1.5:.0f})")
    print(f"   此长度完全在 DeepSeek 128K (约 12.8万 Token) 覆盖范围内。")
    return full_text

# --- 4. 主程序 --- 
def main():
    """
    主函数：执行术语提取流程
    """
    # 配置路径
    input_file = "../data/processed/诡秘之主_final.jsonl"
    output_file = "../data/glossary/project_knowledge_base.json"
    
    print("🔍 启动术语提取Agent...")
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 1. 初始化模型和解析器
    llm = init_llm()
    parser = JsonOutputParser(pydantic_object=KnowledgeGraph)
    
    # 2. 设计提示词
    system_prompt = """
    你是一位资深的奇幻文学翻译总监。你拥有过目不忘的能力，已阅读了《诡秘之主》的前10章全文。
    你的任务是构建一份**“核心术语与世界观指南”**，以确保后续翻译的统一性。
    
    请利用你对全书的理解：
    1. **去重与合并**：同一个实体（如“周明瑞”和“克莱恩”）如果是指向同一人，请在定义中说明，但保留主要称呼作为术语。
    2. **深度理解**：对于“非凡者”、“魔药”等核心设定，不要只看字面意思，要结合上下文总结其在本书中的特殊含义。
    3. **英文命名**：对于人名地名，参考维多利亚时代风格（Victorian Style）；对于专有名词，参考克苏鲁神话（Cthulhu Mythos）风格。
    
    请注意：输出必须严格遵循 JSON 格式，不要包含任何额外的分析文本。
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "这是小说的前 10 章完整内容（共约 2 万字）。请分析并提取核心知识库：\n\n{full_text}\n\n{format_instructions}")
    ])
    
    chain = prompt | llm | parser
    
    # 3. 准备长文本
    full_context = load_full_text(input_file)
    
    # 4. 调用大模型
    print("\n🚀 正在发送请求给 DeepSeek (这可能需要 30-60 秒)...")
    try:
        result = chain.invoke({
            "full_text": full_context,
            "format_instructions": parser.get_format_instructions()
        })
        
        # 5. 结果展示与保存
        print("\n✨ 世界观总结:")
        print(result['world_summary'])
        print(f"\n✨ 提取术语数量: {len(result['terms'])}")
        
        # 打印前几个看看
        for term in result['terms'][:5]:
            print(f"   - {term['term']} ({term['suggested_translation']}): {term['definition']}")

        # 保存为最终知识库
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"\n💾 知识库已保存至 {output_file}")
            
    except Exception as e:
        print(f"❌ 发生错误: {e}")

if __name__ == "__main__":
    main()