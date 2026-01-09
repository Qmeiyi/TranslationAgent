#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TEaR翻译脚本 (Agent 2: Translation & Refinement Agent)
功能：实现翻译-评估-润色的循环流程，包含回译验证
C同学增强版：集成术语一致性强制检查与自动纠正
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from glossary.store import GlossaryStore
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


def _format_glossary(
    glossary: Optional[Dict[str, Any]], text: Optional[str] = None
) -> str:
    """
    格式化术语表用于注入到翻译prompt
    如果提供text，只返回与文本相关的术语（Glossary Injection）
    """
    if not glossary:
        return "No glossary available."

    # 使用GlossaryStore来格式化
    store = GlossaryStore()
    store.glossary = glossary

    if text:
        # 只获取相关术语
        relevant_terms = store.get_relevant_terms(text)
        return store.format_for_prompt(relevant_terms)
    else:
        return store.format_for_prompt()


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
    """
    翻译chunk，包含术语一致性检查与自动纠正
    
    C同学增强：在翻译后检查术语violations，如果发现则触发自动纠正
    """
    if dry_run:
        return {
            "draft": f"[DRAFT] {text[:200]}",
            "back_translation": f"[BACK] {text[:200]}",
            "critique": "DRY_RUN",
            "final_translation": f"[FINAL] {text[:200]}",
            "violations": [],
            "meta": {"model": "dry_run", "time_sec": 0.0},
        }

    # 初始化GlossaryStore用于一致性检查
    store = GlossaryStore()
    if glossary:
        store.glossary = glossary

    # Glossary Injection: 只获取与当前文本相关的术语
    glossary_text = _format_glossary(glossary, text)
    context_text = _format_context(context)

    llm_draft = get_llm(temperature=0.3)
    llm_critique = get_llm(temperature=0.1)
    llm_refine = get_llm(temperature=0.2)
    llm_backtranslate = get_llm(temperature=0.0)
    llm_correct = get_llm(temperature=0.1)  # 用于术语纠正

    draft_chain = DRAFT_PROMPT | llm_draft | StrOutputParser()
    critique_chain = CRITIQUE_PROMPT | llm_critique | StrOutputParser()
    refine_chain = REFINE_PROMPT | llm_refine | StrOutputParser()
    back_chain = BACK_TRANSLATE_PROMPT | llm_backtranslate | StrOutputParser()

    # 术语纠正prompt
    CORRECT_TERMS_PROMPT = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是一位严格的术语检查专家。请修正翻译中的术语不一致问题。",
            ),
            (
                "user",
                "【原文】:\n{original}\n\n【当前翻译】:\n{translation}\n\n"
                "【术语违规列表】:\n{violations_text}\n\n"
                "请修正翻译，确保所有术语都使用正确的翻译。直接输出修正后的英文翻译。",
            ),
        ]
    )
    correct_chain = CORRECT_TERMS_PROMPT | llm_correct | StrOutputParser()

    start_time = time.time()

    # Step 1: 生成初稿
    draft = draft_chain.invoke(
        {"title": title or "", "text": text, "glossary": glossary_text, "context": context_text}
    )

    # Step 2: 回译验证
    back_translation = back_chain.invoke({"text": draft})

    # Step 3: 自我审校
    critique = critique_chain.invoke(
        {
            "original": text,
            "draft": draft,
            "back_translation": back_translation,
            "glossary": glossary_text,
        }
    )

    # Step 4: 根据审校意见优化
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

    # Step 5: C同学增强 - 术语一致性检查与自动纠正
    if glossary:
        term_violations = store.check_violations(text, final_translation)
        if term_violations:
            violations.extend(term_violations)

            # 如果有high或medium严重度的违规，触发自动纠正
            high_severity = [v for v in term_violations if v.get("severity") in ["high", "medium"]]
            if high_severity:
                violations_text = "\n".join(
                    [
                        f"- {v['term']}: 应使用 '{v['expected']}'，但当前翻译中未找到或使用了错误翻译"
                        for v in high_severity
                    ]
                )
                corrected = correct_chain.invoke(
                    {
                        "original": text,
                        "translation": final_translation,
                        "violations_text": violations_text,
                    }
                )
                final_translation = corrected

                # 再次检查纠正后的结果
                remaining_violations = store.check_violations(text, final_translation)
                violations = [v for v in violations if v not in high_severity] + remaining_violations

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
