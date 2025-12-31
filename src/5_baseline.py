#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基线翻译脚本 (Baseline Translator)
功能：不挂载术语表，不使用 TEaR，直接进行单次翻译
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
DEFAULT_OUTPUT_FILE = PROJECT_ROOT / "data" / "output" / "诡秘之主_baseline_result.jsonl"


BASELINE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
你是一位中英翻译专家。请将下面的中文小说翻译成英文。

风格要求：
- 保持原文意思准确
- 使用流畅的英文表达
- 无需特别风格
""",
        ),
        (
            "user",
            "【章节标题】: {title}\n\n【原文内容】:\n{text}\n\n请直接输出英文翻译。",
        ),
    ]
)


def baseline_translate_chunk(
    text: str,
    title: Optional[str] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    if dry_run:
        return {
            "baseline_translation": f"[BASELINE] {text[:200]}",
            "meta": {"model": "dry_run", "time_sec": 0.0},
        }

    llm = get_llm(temperature=0.1)
    chain = BASELINE_PROMPT | llm | StrOutputParser()

    start_time = time.time()
    translation = chain.invoke({"title": title or "", "text": text})
    elapsed = time.time() - start_time

    try:
        model_name = get_llm_settings().model
    except RuntimeError:
        model_name = "unknown"

    return {
        "baseline_translation": translation,
        "meta": {"model": model_name, "time_sec": round(elapsed, 2)},
    }


def main(
    input_file: Path = DEFAULT_INPUT_FILE,
    output_file: Path = DEFAULT_OUTPUT_FILE,
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with input_file.open("r", encoding="utf-8") as fin, output_file.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            chapter = json.loads(line)
            title = chapter.get("title", "")
            text = chapter.get("text", "")

            result = baseline_translate_chunk(text=text, title=title)

            payload = {
                "chapter_index": chapter.get("chapter_index"),
                "title": title,
                "baseline_translation": result["baseline_translation"],
            }

            fout.write(json.dumps(payload, ensure_ascii=False) + "\n")
            fout.flush()


if __name__ == "__main__":
    main()
