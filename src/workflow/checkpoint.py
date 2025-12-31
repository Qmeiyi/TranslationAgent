from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def run_dir(run_id: str) -> Path:
    return _project_root() / "data" / "runs" / run_id


def init_run_dir(run_id: str) -> Dict[str, Path]:
    root = run_dir(run_id)
    states_dir = root / "states"
    chunks_dir = root / "chunks"
    final_dir = root / "final"
    logs_dir = root / "logs"
    glossary_dir = root / "glossary"
    chapters_dir = final_dir / "chapters"

    for path in [states_dir, chunks_dir, final_dir, logs_dir, glossary_dir, chapters_dir]:
        path.mkdir(parents=True, exist_ok=True)

    return {
        "root": root,
        "states": states_dir,
        "chunks": chunks_dir,
        "final": final_dir,
        "logs": logs_dir,
        "glossary": glossary_dir,
        "chapters": chapters_dir,
    }


def save_state(run_id: str, node_name: str, state_dict: Dict[str, Any]) -> Path:
    paths = init_run_dir(run_id)
    state_path = paths["states"] / f"state_{node_name}.json"
    payload = {
        "node_name": node_name,
        "state": state_dict,
    }
    state_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return state_path


def load_latest_state(run_id: str) -> Tuple[str, Dict[str, Any]]:
    states_dir = run_dir(run_id) / "states"
    if not states_dir.exists():
        raise FileNotFoundError(f"No states directory found for run_id={run_id}")

    state_files = list(states_dir.glob("state_*.json"))
    if not state_files:
        raise FileNotFoundError(f"No state files found for run_id={run_id}")

    latest = max(state_files, key=lambda p: p.stat().st_mtime)
    payload = json.loads(latest.read_text(encoding="utf-8"))
    node_name = payload.get("node_name") or latest.stem.replace("state_", "")
    state = payload.get("state", payload)
    return node_name, state


def chunk_exists(run_id: str, chapter_id: int, chunk_id: str) -> bool:
    chunk_path = run_dir(run_id) / "chunks" / str(chapter_id) / f"{chunk_id}.json"
    return chunk_path.exists()


def save_chunk_output(run_id: str, chapter_id: int, chunk_id: str, obj: Any) -> Path:
    chunk_dir = run_dir(run_id) / "chunks" / str(chapter_id)
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunk_path = chunk_dir / f"{chunk_id}.json"
    chunk_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return chunk_path


def load_chunk_output(run_id: str, chapter_id: int, chunk_id: str) -> Any:
    chunk_path = run_dir(run_id) / "chunks" / str(chapter_id) / f"{chunk_id}.json"
    return json.loads(chunk_path.read_text(encoding="utf-8"))


def append_log(run_id: str, message: str) -> Path:
    logs_dir = init_run_dir(run_id)["logs"]
    log_path = logs_dir / "run.log"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with log_path.open("a", encoding="utf-8") as f:
        lines = message.splitlines() or [""]
        for line in lines:
            f.write(f"[{timestamp}] {line}\n")
    return log_path
