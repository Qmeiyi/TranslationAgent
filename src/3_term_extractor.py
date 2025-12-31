#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
术语提取脚本 (Agent 1: Terminology & Entity Agent)
功能：识别并规范化小说中的术语、实体和文化负载词
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional, Union

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from utils.config import get_llm

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class GlossaryEntry(BaseModel):
    term: str = Field(description="源术语")
    type: str = Field(description="类型，例如 NE:person, NE:location, NE:org")
    final: str = Field(description="最终译名")
    aliases: List[str] = Field(default_factory=list)
    senses: List[str] = Field(default_factory=list)


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

    llm = get_llm(temperature=0.1, max_tokens=4000)
    parser = JsonOutputParser(pydantic_object=Glossary)

    system_prompt = (
        "你是一位资深的奇幻文学翻译总监，熟悉《诡秘之主》世界观。"
        "请从文本中提取核心术语，返回 JSON 格式术语表。"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "user",
                "这是小说内容片段。请输出术语表 JSON：\n\n{full_text}\n\n{format_instructions}",
            ),
        ]
    )

    chain = prompt | llm | parser
    glossary = chain.invoke(
        {
            "full_text": full_text,
            "format_instructions": parser.get_format_instructions(),
        }
    )

    if "version" not in glossary:
        glossary["version"] = 1

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
