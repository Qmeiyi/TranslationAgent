#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一 CLI 入口：LangGraph Workflow + checkpoint/resume
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from workflow.checkpoint import append_log, init_run_dir, load_latest_state
from workflow.graph import build_workflow, next_node_after


def _parse_chapter_range(value: Optional[str]) -> Optional[Dict[str, int]]:
    if not value:
        return None
    if "-" in value:
        start_str, end_str = value.split("-", 1)
        return {"start": int(start_str), "end": int(end_str)}
    idx = int(value)
    return {"start": idx, "end": idx}


def _make_run_id(input_path: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    digest = hashlib.sha1(f"{input_path}-{timestamp}".encode("utf-8")).hexdigest()[:6]
    return f"{timestamp}_{digest}"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TranslationAgent LangGraph Workflow")
    parser.add_argument("--input_path", help="输入文本 (jsonl 或 txt)")
    parser.add_argument("--chapters", help="章节范围，例如 1-2 或 3")
    parser.add_argument("--run_id", help="指定 run_id (可选)")
    parser.add_argument("--resume", help="从已有 run_id 恢复")
    parser.add_argument("--dry_run", action="store_true", help="仅跑流程，不调用 LLM")
    parser.add_argument("--max_chapters", type=int, help="限制章节数量")
    parser.add_argument("--max_chunks_per_chapter", type=int, help="限制每章 chunk 数")
    parser.add_argument("--chunk_size", type=int, default=1200, help="chunk 目标长度(字符)")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if not args.resume and not args.input_path:
        parser.error("--input_path is required unless --resume is provided")

    if args.resume:
        run_id = args.resume
        init_run_dir(run_id)
        append_log(run_id, f"[main] resume requested run_id={run_id}")
        if args.input_path:
            append_log(run_id, "[main] --input_path ignored because --resume was used")

        last_node, state = load_latest_state(run_id)
        resume_from = next_node_after(last_node)

        state["run_id"] = run_id
        state["resume_from"] = resume_from
        state.setdefault("config", {})
        if args.dry_run:
            state["config"]["dry_run"] = True
        append_log(
            run_id,
            f"[main] resuming from {resume_from} (last node: {last_node})",
        )
    else:
        input_path = str(Path(args.input_path).resolve())
        run_id = args.run_id or _make_run_id(input_path)
        init_run_dir(run_id)
        append_log(run_id, f"[main] new run run_id={run_id}")

        chapter_range = _parse_chapter_range(args.chapters)
        config = {
            "dry_run": bool(args.dry_run),
            "max_chapters": args.max_chapters,
            "max_chunks_per_chapter": args.max_chunks_per_chapter,
            "chunk_size": args.chunk_size,
        }

        state = {
            "run_id": run_id,
            "input_path": input_path,
            "chapter_range": chapter_range,
            "config": config,
            "progress": {},
            "resume_from": "load_text",
        }

    workflow = build_workflow()
    try:
        workflow.invoke(state)
    except KeyboardInterrupt:
        append_log(run_id, "[main] interrupted by user")
        print(f"Interrupted. You can resume with --resume {run_id}.")
        return 130
    except Exception as exc:
        append_log(run_id, f"[main] error: {exc}")
        append_log(run_id, traceback.format_exc().strip())
        raise

    print(f"Run completed. run_id={run_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
