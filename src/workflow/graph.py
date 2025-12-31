from __future__ import annotations

import importlib.util
import json
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from workflow.checkpoint import (
    append_log,
    chunk_exists,
    init_run_dir,
    load_chunk_output,
    save_chunk_output,
    save_state,
)


class WorkflowState(TypedDict, total=False):
    run_id: str
    input_path: str
    chapters: List[Dict[str, Any]]
    chunks: List[Dict[str, Any]]
    glossary: Dict[str, Any]
    outputs: Dict[str, Any]
    progress: Dict[str, Any]
    config: Dict[str, Any]
    chapter_range: Dict[str, int]
    resume_from: str


SegmenterFn = Callable[
    [List[Dict[str, Any]], int, Optional[Callable[..., Dict[str, Any]]], Optional[int]],
    List[Dict[str, Any]],
]
ContextBuilderFn = Callable[[Dict[str, Any], str, int, List[Dict[str, Any]]], Dict[str, Any]]
GlossaryLoaderFn = Callable[[List[Dict[str, Any]], bool, Path], Dict[str, Any]]
TranslatorFn = Callable[
    [str, Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[str], bool],
    Dict[str, Any],
]

NODE_ORDER = [
    "load_text",
    "segment",
    "load_or_extract_glossary",
    "translate_chunks",
    "merge_outputs",
]


_TERM_MODULE: Optional[Any] = None
_TRANSLATOR_MODULE: Optional[Any] = None


def _load_module(module_filename: str, module_name: str) -> Any:
    module_path = Path(__file__).resolve().parents[1] / module_filename
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ImportError(f"Unable to load module from {module_path}")
    spec.loader.exec_module(module)
    return module


def _get_term_module() -> Any:
    global _TERM_MODULE
    if _TERM_MODULE is None:
        _TERM_MODULE = _load_module("3_term_extractor.py", "term_extractor")
    return _TERM_MODULE


def _get_translator_module() -> Any:
    global _TRANSLATOR_MODULE
    if _TRANSLATOR_MODULE is None:
        _TRANSLATOR_MODULE = _load_module("4_translator_tear.py", "translator_tear")
    return _TRANSLATOR_MODULE


def _default_context_builder(
    chapter: Dict[str, Any],
    chunk_text: str,
    chunk_index: int,
    previous_chunks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "chapter_title": chapter.get("title"),
        "chunk_index": chunk_index,
        "prev_chunk_tail": previous_chunks[-1]["text"][-200:]
        if previous_chunks
        else "",
    }


def _default_segmenter(
    chapters: List[Dict[str, Any]],
    max_chars: int,
    context_builder: Optional[ContextBuilderFn] = None,
    max_chunks_per_chapter: Optional[int] = None,
) -> List[Dict[str, Any]]:
    builder = context_builder or _default_context_builder
    chunks: List[Dict[str, Any]] = []

    for chapter in chapters:
        chapter_id = chapter.get("chapter_id")
        text = chapter.get("text", "")
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        current_parts: List[str] = []
        current_len = 0
        chunk_index = 1
        chapter_chunks: List[Dict[str, Any]] = []

        def flush_chunk() -> None:
            nonlocal chunk_index, current_parts, current_len, chapter_chunks
            if not current_parts:
                return
            chunk_text = "\n\n".join(current_parts)
            chunk_id = f"{chapter_id}-{chunk_index:03d}"
            context = builder(chapter, chunk_text, chunk_index, chapter_chunks)
            chunk = {
                "chapter_id": chapter_id,
                "chunk_id": chunk_id,
                "text": chunk_text,
                "context": context,
            }
            chunks.append(chunk)
            chapter_chunks.append(chunk)
            chunk_index += 1
            current_parts = []
            current_len = 0

        for paragraph in paragraphs:
            if current_parts and current_len + len(paragraph) + 2 > max_chars:
                flush_chunk()
                if max_chunks_per_chapter and chunk_index > max_chunks_per_chapter:
                    break
            current_parts.append(paragraph)
            current_len += len(paragraph) + 2

        if not max_chunks_per_chapter or chunk_index <= max_chunks_per_chapter:
            flush_chunk()

    return chunks


def _normalize_glossary(glossary: Dict[str, Any]) -> Dict[str, Any]:
    if "entries" in glossary:
        return glossary

    if "terms" in glossary:
        entries = []
        for term in glossary.get("terms", []):
            entries.append(
                {
                    "term": term.get("term"),
                    "type": term.get("category"),
                    "final": term.get("suggested_translation"),
                    "aliases": [],
                    "senses": [],
                }
            )
        normalized = {
            "version": glossary.get("version", 1),
            "entries": entries,
        }
        if glossary.get("world_summary"):
            normalized["world_summary"] = glossary.get("world_summary")
        return normalized

    return {"version": glossary.get("version", 1), "entries": []}


def _default_glossary_loader(
    chapters: List[Dict[str, Any]],
    dry_run: bool,
    glossary_path: Path,
) -> Dict[str, Any]:
    if glossary_path.exists():
        return _normalize_glossary(
            json.loads(glossary_path.read_text(encoding="utf-8"))
        )

    project_glossary = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "glossary"
        / "project_knowledge_base.json"
    )
    if project_glossary.exists():
        return _normalize_glossary(
            json.loads(project_glossary.read_text(encoding="utf-8"))
        )

    term_module = _get_term_module()
    glossary = term_module.extract_terms(chapters, dry_run=dry_run)
    return _normalize_glossary(glossary)


def _default_translator(
    text: str,
    glossary: Optional[Dict[str, Any]],
    context: Optional[Dict[str, Any]],
    title: Optional[str],
    dry_run: bool,
) -> Dict[str, Any]:
    translator_module = _get_translator_module()
    return translator_module.translate_chunk_tear(
        text=text, glossary=glossary, context=context, title=title, dry_run=dry_run
    )


def _load_input_text(state: WorkflowState) -> List[Dict[str, Any]]:
    input_path = Path(state["input_path"])
    if input_path.suffix.lower() == ".jsonl":
        chapters: List[Dict[str, Any]] = []
        with input_path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f, start=1):
                data = json.loads(line)
                chapter_id = data.get("chapter_index") or data.get("chapter_id") or idx
                chapters.append(
                    {
                        "chapter_id": int(chapter_id),
                        "title": data.get("title", ""),
                        "text": data.get("text", ""),
                    }
                )
    else:
        text = input_path.read_text(encoding="utf-8")
        chapters = [
            {
                "chapter_id": 1,
                "title": input_path.stem,
                "text": text,
            }
        ]

    chapter_range = state.get("chapter_range")
    if chapter_range and len(chapters) > 1:
        start = chapter_range.get("start")
        end = chapter_range.get("end")
        if start is not None and end is not None:
            chapters = [
                c
                for c in chapters
                if int(c["chapter_id"]) >= start and int(c["chapter_id"]) <= end
            ]

    max_chapters = state.get("config", {}).get("max_chapters")
    if max_chapters:
        chapters = chapters[:max_chapters]

    return chapters


def _log_node_start(run_id: str, node_name: str) -> None:
    append_log(run_id, f"[{node_name}] start")


def _log_node_end(run_id: str, node_name: str, elapsed: float, details: str = "") -> None:
    suffix = f" {details}" if details else ""
    append_log(run_id, f"[{node_name}] end ({elapsed:.2f}s){suffix}")


def load_text_node(state: WorkflowState) -> WorkflowState:
    run_id = state["run_id"]
    _log_node_start(run_id, "load_text")
    start = time.time()

    try:
        chapters = _load_input_text(state)
        progress = dict(state.get("progress", {}))
        progress["load_text"] = {
            "chapters": len(chapters),
        }

        next_state: WorkflowState = {
            **state,
            "chapters": chapters,
            "progress": progress,
        }

        state_path = save_state(run_id, "load_text", next_state)
        _log_node_end(
            run_id,
            "load_text",
            time.time() - start,
            f"chapters={len(chapters)} state={state_path}",
        )
        return next_state
    except Exception as exc:
        append_log(run_id, f"[load_text] error: {exc}")
        append_log(run_id, traceback.format_exc().strip())
        raise


def segment_node(
    state: WorkflowState,
    segmenter: SegmenterFn,
    context_builder: Optional[ContextBuilderFn],
) -> WorkflowState:
    run_id = state["run_id"]
    _log_node_start(run_id, "segment")
    start = time.time()

    try:
        config = state.get("config", {})
        max_chars = config.get("chunk_size", 1200)
        max_chunks = config.get("max_chunks_per_chapter")

        chunks = segmenter(
            state.get("chapters", []),
            max_chars,
            context_builder,
            max_chunks,
        )

        progress = dict(state.get("progress", {}))
        progress["segment"] = {"chunks": len(chunks)}

        next_state: WorkflowState = {
            **state,
            "chunks": chunks,
            "progress": progress,
        }

        state_path = save_state(run_id, "segment", next_state)
        _log_node_end(
            run_id,
            "segment",
            time.time() - start,
            f"chunks={len(chunks)} state={state_path}",
        )
        return next_state
    except Exception as exc:
        append_log(run_id, f"[segment] error: {exc}")
        append_log(run_id, traceback.format_exc().strip())
        raise


def load_or_extract_glossary_node(
    state: WorkflowState, glossary_loader: GlossaryLoaderFn
) -> WorkflowState:
    run_id = state["run_id"]
    _log_node_start(run_id, "load_or_extract_glossary")
    start = time.time()

    try:
        paths = init_run_dir(run_id)
        glossary_path = paths["glossary"] / "glossary.json"
        dry_run = state.get("config", {}).get("dry_run", False)

        glossary = glossary_loader(state.get("chapters", []), dry_run, glossary_path)
        glossary_path.write_text(
            json.dumps(glossary, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        progress = dict(state.get("progress", {}))
        progress["load_or_extract_glossary"] = {
            "entries": len(glossary.get("entries", []))
        }

        next_state: WorkflowState = {
            **state,
            "glossary": glossary,
            "progress": progress,
        }

        state_path = save_state(run_id, "load_or_extract_glossary", next_state)
        _log_node_end(
            run_id,
            "load_or_extract_glossary",
            time.time() - start,
            f"entries={len(glossary.get('entries', []))} glossary={glossary_path} state={state_path}",
        )
        return next_state
    except Exception as exc:
        append_log(run_id, f"[load_or_extract_glossary] error: {exc}")
        append_log(run_id, traceback.format_exc().strip())
        raise


def translate_chunks_node(state: WorkflowState, translator: TranslatorFn) -> WorkflowState:
    run_id = state["run_id"]
    _log_node_start(run_id, "translate_chunks")
    start = time.time()

    chunks = state.get("chunks", [])
    glossary = state.get("glossary")
    dry_run = state.get("config", {}).get("dry_run", False)
    chapter_titles = {
        chapter["chapter_id"]: chapter.get("title", "")
        for chapter in state.get("chapters", [])
    }

    translated = 0
    skipped = 0
    for chunk in chunks:
        chapter_id = chunk["chapter_id"]
        chunk_id = chunk["chunk_id"]

        if chunk_exists(run_id, chapter_id, chunk_id):
            skipped += 1
            continue

        try:
            result = translator(
                text=chunk["text"],
                glossary=glossary,
                context=chunk.get("context"),
                title=chapter_titles.get(chapter_id, ""),
                dry_run=dry_run,
            )
        except Exception as exc:
            append_log(
                run_id,
                f"[translate_chunks] error chunk_id={chunk_id} chapter_id={chapter_id}: {exc}",
            )
            append_log(run_id, traceback.format_exc().strip())
            raise

        payload = {
            "chapter_id": chapter_id,
            "chunk_id": chunk_id,
            "source": chunk["text"],
            "draft": result.get("draft"),
            "back_translation": result.get("back_translation"),
            "critique": result.get("critique"),
            "final_translation": result.get("final_translation"),
            "violations": result.get("violations", []),
            "meta": result.get("meta", {}),
        }

        save_chunk_output(run_id, chapter_id, chunk_id, payload)
        translated += 1

    progress = dict(state.get("progress", {}))
    progress["translate_chunks"] = {
        "translated": translated,
        "skipped": skipped,
    }

    next_state: WorkflowState = {
        **state,
        "progress": progress,
    }

    state_path = save_state(run_id, "translate_chunks", next_state)
    _log_node_end(
        run_id,
        "translate_chunks",
        time.time() - start,
        f"translated={translated} skipped={skipped} chunks_dir={init_run_dir(run_id)['chunks']} state={state_path}",
    )
    return next_state


def merge_outputs_node(state: WorkflowState) -> WorkflowState:
    run_id = state["run_id"]
    _log_node_start(run_id, "merge_outputs")
    start = time.time()

    try:
        chunks = state.get("chunks", [])
        paths = init_run_dir(run_id)
        chapters_dir = paths["chapters"]
        book_path = paths["final"] / "book_merged_zh.txt"

        chapters_map: Dict[int, List[Dict[str, Any]]] = {}
        for chunk in chunks:
            chapters_map.setdefault(chunk["chapter_id"], []).append(chunk)

        merged_book_parts: List[str] = []
        chapter_outputs: Dict[int, str] = {}

        for chapter_id in sorted(chapters_map.keys()):
            ordered_chunks = sorted(
                chapters_map[chapter_id], key=lambda c: c["chunk_id"]
            )
            translations: List[str] = []
            for chunk in ordered_chunks:
                data = load_chunk_output(run_id, chapter_id, chunk["chunk_id"])
                translations.append(data.get("final_translation", ""))
            chapter_text = "\n".join(translations)
            chapter_path = chapters_dir / f"chapter_{chapter_id}_zh.txt"
            chapter_path.write_text(chapter_text, encoding="utf-8")
            merged_book_parts.append(chapter_text)
            chapter_outputs[chapter_id] = str(chapter_path)

        book_path.write_text("\n".join(merged_book_parts), encoding="utf-8")

        outputs = {
            "book": str(book_path),
            "chapters": chapter_outputs,
        }

        progress = dict(state.get("progress", {}))
        progress["merge_outputs"] = {"chapters": len(chapter_outputs)}

        next_state: WorkflowState = {
            **state,
            "outputs": outputs,
            "progress": progress,
        }

        state_path = save_state(run_id, "merge_outputs", next_state)
        _log_node_end(
            run_id,
            "merge_outputs",
            time.time() - start,
            f"book={book_path} chapters_dir={chapters_dir} state={state_path}",
        )
        return next_state
    except Exception as exc:
        append_log(run_id, f"[merge_outputs] error: {exc}")
        append_log(run_id, traceback.format_exc().strip())
        raise


def route_node(state: WorkflowState) -> WorkflowState:
    return state


def route_decider(state: WorkflowState) -> str:
    return state.get("resume_from", "load_text")


def build_workflow(
    segmenter: Optional[SegmenterFn] = None,
    context_builder: Optional[ContextBuilderFn] = None,
    glossary_loader: Optional[GlossaryLoaderFn] = None,
    translator: Optional[TranslatorFn] = None,
):
    """Build the LangGraph workflow with optional hook overrides."""
    segmenter_fn = segmenter or _default_segmenter
    glossary_loader_fn = glossary_loader or _default_glossary_loader
    translator_fn = translator or _default_translator

    graph = StateGraph(WorkflowState)

    graph.add_node("route", route_node)
    graph.add_node("load_text", load_text_node)
    graph.add_node(
        "segment",
        lambda state: segment_node(state, segmenter_fn, context_builder),
    )
    graph.add_node(
        "load_or_extract_glossary",
        lambda state: load_or_extract_glossary_node(state, glossary_loader_fn),
    )
    graph.add_node(
        "translate_chunks",
        lambda state: translate_chunks_node(state, translator_fn),
    )
    graph.add_node("merge_outputs", merge_outputs_node)

    graph.set_entry_point("route")
    graph.add_conditional_edges(
        "route",
        route_decider,
        {
            "load_text": "load_text",
            "segment": "segment",
            "load_or_extract_glossary": "load_or_extract_glossary",
            "translate_chunks": "translate_chunks",
            "merge_outputs": "merge_outputs",
            "end": END,
        },
    )

    graph.add_edge("load_text", "segment")
    graph.add_edge("segment", "load_or_extract_glossary")
    graph.add_edge("load_or_extract_glossary", "translate_chunks")
    graph.add_edge("translate_chunks", "merge_outputs")
    graph.add_edge("merge_outputs", END)

    return graph.compile()


def next_node_after(node_name: str) -> str:
    if node_name not in NODE_ORDER:
        return "load_text"
    idx = NODE_ORDER.index(node_name)
    if idx + 1 >= len(NODE_ORDER):
        return "end"
    return NODE_ORDER[idx + 1]
