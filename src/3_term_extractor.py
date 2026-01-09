#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
术语提取脚本 (Agent 1: Terminology & Entity Agent)
功能：识别并规范化小说中的术语、实体和文化负载词
增强版：支持多类别术语提取，包含证据片段和候选翻译
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from utils.config import get_llm

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TermExtractionResult(BaseModel):
    """单个术语提取结果（用于提取阶段）"""
    term: str = Field(description="原文术语")
    type: str = Field(
        description="类型：NE:person, NE:location, NE:org, NE:work, slang, domain_term, culture_loaded"
    )
    meaning: Optional[str] = Field(default=None, description="术语含义/定义")
    candidates: List[str] = Field(default_factory=list, description="候选翻译列表")
    evidence_span: str = Field(description="该术语首次出现或最关键的原文片段引用")
    chapter_id: Optional[int] = Field(default=None, description="章节ID")
    chunk_id: Optional[str] = Field(default=None, description="chunk ID")


class TermExtractionOutput(BaseModel):
    """术语提取输出（批量）"""
    terms: List[TermExtractionResult]


class GlossaryEntry(BaseModel):
    """Glossary中的条目（用于持久化）"""
    term: str = Field(description="源术语")
    type: str = Field(description="类型，例如 NE:person, NE:location, NE:org, slang, domain_term")
    final: str = Field(description="最终译名")
    aliases: List[str] = Field(default_factory=list, description="别名列表")
    senses: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="多义词义项列表，每个义项包含sense_id、context_signature、final等"
    )
    candidates: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="候选翻译列表，包含translation、score、source等"
    )
    evidence_span: Optional[str] = Field(default=None, description="证据片段")
    score: Optional[float] = Field(default=None, description="评分")


class Glossary(BaseModel):
    version: int = Field(default=1, description="术语表版本")
    entries: List[GlossaryEntry]
    world_summary: Optional[str] = Field(default=None, description="世界观摘要")


def _stringify_chapters(chapters: Iterable[dict]) -> str:
    parts: List[str] = []
    for chapter in chapters:
        chapter_id = chapter.get("chapter_id") or chapter.get("chapter_index") or "?"
        title = chapter.get("title") or ""
        text = chapter.get("text") or ""
        parts.append(f"[Chapter {chapter_id} {title}]\n{text}")
    return "\n\n".join(parts)


def load_full_text(input_path: Path) -> str:
    if input_path.suffix.lower() == ".jsonl":
        chapters = []
        with input_path.open("r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                chapters.append(data)
        return _stringify_chapters(chapters)

    return input_path.read_text(encoding="utf-8")


def extract_terms(
    chapters_or_text: Union[str, Path, Iterable[dict]],
    output_path: Optional[Path] = None,
    dry_run: bool = False,
    max_chars: int = 60000,
) -> dict:
    if isinstance(chapters_or_text, Path):
        full_text = load_full_text(chapters_or_text)
    elif isinstance(chapters_or_text, str):
        full_text = chapters_or_text
    else:
        full_text = _stringify_chapters(chapters_or_text)

    if max_chars and len(full_text) > max_chars:
        full_text = full_text[:max_chars]

    if dry_run:
        glossary = {"version": 1, "entries": [], "world_summary": ""}
        if output_path:
            output_path.write_text(
                json.dumps(glossary, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        return glossary

    llm = get_llm(temperature=0.1, max_tokens=6000)
    
    # 使用TermExtractionOutput进行提取
    extraction_parser = JsonOutputParser(pydantic_object=TermExtractionOutput)
    
    system_prompt = """你是一位资深的奇幻文学翻译总监，熟悉《诡秘之主》世界观。
你的任务是从文本中提取核心术语，包括：
1. **命名实体（NE）**：
   - NE:person（人名，如"克莱恩·莫雷蒂"）
   - NE:location（地名，如"廷根市"）
   - NE:org（组织名，如"值夜者"）
   - NE:work（作品名、概念名，如"赫密斯文"）

2. **俚语（slang）**：小说中的特殊表达、俚语

3. **领域术语（domain_term）**：世界观相关的专业术语，如"非凡者"、"魔药"、"序列"

4. **文化负载词（culture_loaded）**：具有特定文化内涵的词汇

对于每个术语，请提供：
- term: 原文术语
- type: 类型（必须从上述类别中选择）
- meaning: 术语的含义/定义
- candidates: 候选翻译列表（至少2-3个选项）
- evidence_span: 该术语首次出现或最关键的原文片段（50-100字）

请确保提取的术语是重要的、需要全书一致的专有名词和关键概念。"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "user",
                "这是小说内容片段。请提取术语并返回JSON格式：\n\n{full_text}\n\n{format_instructions}",
            ),
        ]
    )

    chain = prompt | llm | extraction_parser
    extraction_result = chain.invoke(
        {
            "full_text": full_text,
            "format_instructions": extraction_parser.get_format_instructions(),
        }
    )

    # 将提取结果转换为Glossary格式
    entries = []
    for term_data in extraction_result.get("terms", []):
        entry = GlossaryEntry(
            term=term_data.get("term", ""),
            type=term_data.get("type", ""),
            final=term_data.get("candidates", [""])[0] if term_data.get("candidates") else "",
            candidates=[
                {
                    "translation": cand,
                    "score": 1.0 - (i * 0.1),  # 第一个候选分数最高
                    "source": "extraction"
                }
                for i, cand in enumerate(term_data.get("candidates", []))
            ],
            evidence_span=term_data.get("evidence_span"),
        )
        entries.append(entry)

    glossary = {
        "version": 1,
        "entries": [entry.model_dump() for entry in entries],
        "world_summary": "",
    }

    if output_path:
        output_path.write_text(
            json.dumps(glossary, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    return glossary


def main() -> None:
    input_file = PROJECT_ROOT / "data" / "processed" / "诡秘之主_final.jsonl"
    output_file = PROJECT_ROOT / "data" / "glossary" / "project_knowledge_base.json"

    print("🔍 启动术语提取Agent...")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    glossary = extract_terms(input_file, output_path=output_file)

    print(f"✨ 提取术语数量: {len(glossary.get('entries', []))}")
    print(f"💾 知识库已保存至 {output_file}")


if __name__ == "__main__":
    main()
