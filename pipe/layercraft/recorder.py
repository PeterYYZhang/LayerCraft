"""Run artifact persistence."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

from .image_models import FLUX_BG_MODEL, FLUX_FILL_MODEL, GPT_IMAGE_MODEL
from .llm import DEFAULT_LLM_MODEL


def safe_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in value.lower()).strip("_")
    return cleaned[:60] or "object"


def prompt_output_filename(prompt: str) -> str:
    stem = prompt[:50].strip()
    stem = stem.replace("/", "_").replace("\\", "_").replace("\x00", "")
    return f"{stem or 'image'}.png"


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)


def _summary(value: Any) -> Any:
    if isinstance(value, dict):
        summary: dict[str, Any] = {}
        for key, item in value.items():
            if key.endswith("image_id") or key in {"ok", "issues", "final_path"}:
                summary[key] = _jsonable(item)
        return summary or _jsonable(value)
    return _jsonable(value)


class RunRecorder:
    """Writes every run artifact into a stable per-run directory."""

    def __init__(self, prompt: str, runs_root: Path | None = None) -> None:
        self.prompt = prompt
        self.started_at = time.perf_counter()
        root = runs_root or Path(__file__).resolve().parents[1] / "runs"
        short_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8]
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_id = f"{timestamp}_{short_hash}"
        self.run_dir = root / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self._trace_path = self.run_dir / "trace.jsonl"
        self._registered: dict[str, str] = {}
        self._manifest: dict[str, Any] = {
            "run_id": self.run_id,
            "prompt": prompt,
            "status": "running",
            "models": {
                "llm": DEFAULT_LLM_MODEL,
                "image": GPT_IMAGE_MODEL,
                "flux_background": FLUX_BG_MODEL,
                "flux_fill": FLUX_FILL_MODEL,
            },
            "total_tokens": None,
            "total_latency_ms": None,
            "final_path": None,
        }
        (self.run_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
        self.write_manifest()

    def path(self, relative: str) -> Path:
        target = self.run_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def save_json(self, relative: str, payload: Any) -> Path:
        target = self.path(relative)
        target.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
        return target

    def save_image(
        self,
        image_id: str,
        image: Image.Image,
        *,
        artifact_name: str | None = None,
    ) -> Path:
        relative = artifact_name or f"images/{image_id}.png"
        target = self.path(relative)
        image.save(target)
        self._registered[image_id] = str(target)
        return target

    def copy_to_final(
        self,
        source_image: Image.Image,
        *,
        output_root: Path | None = None,
    ) -> Path:
        run_target = self.path("final.png")
        source_image.save(run_target)
        target = run_target
        if output_root is not None:
            output_root.mkdir(parents=True, exist_ok=True)
            target = output_root / prompt_output_filename(self.prompt)
            source_image.save(target)
        self._manifest["final_path"] = str(target.resolve())
        return target

    def copy_existing_to_final(self, source: Path) -> Path:
        target = self.path("final.png")
        shutil.copyfile(source, target)
        self._manifest["final_path"] = str(target.resolve())
        return target

    def record_tool_call(
        self,
        *,
        agent: str,
        tool: str,
        args: dict[str, Any],
        result: Any,
        latency_ms: int,
    ) -> None:
        row = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "agent": agent,
            "tool": tool,
            "args": _jsonable(args),
            "result_summary": _summary(result),
            "latency_ms": latency_ms,
        }
        with self._trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")

    def write_manifest(self, *, status: str | None = None) -> None:
        if status is not None:
            self._manifest["status"] = status
        self._manifest["total_latency_ms"] = int((time.perf_counter() - self.started_at) * 1000)
        self.save_json("manifest.json", self._manifest)

    def finalize(self, *, status: str) -> None:
        self.write_manifest(status=status)
