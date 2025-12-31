#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TEaR翻译脚本 (Agent 2: Translation & Refinement Agent)
功能：实现翻译-评估-润色的循环流程，包含回译验证
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from utils.config import get_llm, get_llm_settings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "诡秘之主_final.jsonl"
DEFAULT_GLOSSARY_FILE = PROJECT_ROOT / "data" / "glossary" / "project_knowledge_base.json"
DEFAULT_OUTPUT_FILE = PROJECT_ROOT / "data" / "output" / "诡秘之主_tear_result.jsonl"


DRAFT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
你是一位精通中英翻译的小说家，擅长“维多利亚奇幻”与“克苏鲁神话”风格。
你的任务是将中文小说翻译成英文。

风格要求:
1. 氛围保持神秘、压抑、复古的英伦风。
2. 用词偏古典，避免现代俚语。
3. 忠实原文叙事节奏。

{glossary}
{context}
""",
        ),
        (
            "user",
            "【章节标题】: {title}\n\n【原文内容】:\n{text}\n\n请直接输出英文初稿。",
        ),
    ]
)

CRITIQUE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
你是一位严苛的翻译审校专家。请检查英文草稿的术语一致性、漏译错译、风格违和。

{glossary}
""",
        ),
        (
            "user",
            "【原文】:\n{original}\n\n【英文初稿】:\n{draft}\n\n【回译中文】:\n{back_translation}\n\n"
            "请列出具体修改建议（如果翻译完美，请回复 'PASS'）。",
        ),
    ]
)

REFINE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
你是一位追求完美的文学编辑。请根据审校意见润色译文。

{glossary}
""",
        ),
        (
            "user",
            "【原文】:\n{original}\n\n【初稿】:\n{draft}\n\n【审校意见】:\n{critique}\n\n"
            "请输出最终版本的英文译文。",
        ),
    ]
)

BACK_TRANSLATE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一位专业翻译专家。请将下面英文准确回译成中文。"),
        ("user", "{text}"),
    ]
)


def _format_glossary(glossary: Optional[Dict[str, Any]]) -> str:
    if not glossary:
        return "No glossary available."

    entries = glossary.get("entries")
    if entries is None and glossary.get("terms"):
        entries = [
            {
                "term": term.get("term"),
                "final": term.get("suggested_translation"),
                "type": term.get("category") or "",
            }
            for term in glossary.get("terms", [])
        ]

    lines = ["## Glossary (Strict):"]
    for entry in entries or []:
        term = entry.get("term", "")
        final = entry.get("final", "")
        term_type = entry.get("type", "")
        lines.append(f"- {term}: {final} {term_type}".strip())

    world_summary = glossary.get("world_summary")
    if world_summary:
        lines.append("\n## World Summary:\n" + world_summary)

    return "\n".join(lines)


def _format_context(context: Optional[Dict[str, Any]]) -> str:
    if not context:
        return ""
    return "## Context:\n" + json.dumps(context, ensure_ascii=False, indent=2)


def translate_chunk_tear(
    text: str,
    glossary: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
    title: Optional[str] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    if dry_run:
        return {
            "draft": f"[DRAFT] {text[:200]}",
            "back_translation": f"[BACK] {text[:200]}",
            "critique": "DRY_RUN",
            "final_translation": f"[FINAL] {text[:200]}",
            "violations": [],
            "meta": {"model": "dry_run", "time_sec": 0.0},
        }

    glossary_text = _format_glossary(glossary)
    context_text = _format_context(context)

    llm_draft = get_llm(temperature=0.3)
    llm_critique = get_llm(temperature=0.1)
    llm_refine = get_llm(temperature=0.2)
    llm_backtranslate = get_llm(temperature=0.0)

    draft_chain = DRAFT_PROMPT | llm_draft | StrOutputParser()
    critique_chain = CRITIQUE_PROMPT | llm_critique | StrOutputParser()
    refine_chain = REFINE_PROMPT | llm_refine | StrOutputParser()
    back_chain = BACK_TRANSLATE_PROMPT | llm_backtranslate | StrOutputParser()

    start_time = time.time()

    draft = draft_chain.invoke(
        {"title": title or "", "text": text, "glossary": glossary_text, "context": context_text}
    )
    back_translation = back_chain.invoke({"text": draft})
    critique = critique_chain.invoke(
        {
            "original": text,
            "draft": draft,
            "back_translation": back_translation,
            "glossary": glossary_text,
        }
    )

    final_translation = draft
    violations = []
    if "PASS" not in critique and len(critique.strip()) > 10:
        final_translation = refine_chain.invoke(
            {
                "original": text,
                "draft": draft,
                "critique": critique,
                "glossary": glossary_text,
            }
        )
        violations.append("needs_refine")

    elapsed = time.time() - start_time
    try:
        model_name = get_llm_settings().model
    except RuntimeError:
        model_name = "unknown"

    return {
        "draft": draft,
        "back_translation": back_translation,
        "critique": critique,
        "final_translation": final_translation,
        "violations": violations,
        "meta": {"model": model_name, "time_sec": round(elapsed, 2)},
    }


def _load_glossary(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"version": 1, "entries": []}
    return json.loads(path.read_text(encoding="utf-8"))


def process_translation(
    input_file: Path = DEFAULT_INPUT_FILE,
    glossary_file: Path = DEFAULT_GLOSSARY_FILE,
    output_file: Path = DEFAULT_OUTPUT_FILE,
) -> None:
    glossary = _load_glossary(glossary_file)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with input_file.open("r", encoding="utf-8") as fin, output_file.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            chapter = json.loads(line)
            title = chapter.get("title", "")
            text = chapter.get("text", "")

            result = translate_chunk_tear(text=text, glossary=glossary, title=title)

            payload = {
                "chapter_index": chapter.get("chapter_index"),
                "title": title,
                "draft": result["draft"],
                "back_translation": result["back_translation"],
                "critique": result["critique"],
                "final_translation": result["final_translation"],
            }
            fout.write(json.dumps(payload, ensure_ascii=False) + "\n")
            fout.flush()


if __name__ == "__main__":
    process_translation()
