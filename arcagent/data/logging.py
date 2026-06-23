from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
import json
import time
from typing import Any

import numpy as np


class MetricLogger:
    """Append-only JSONL logger used by trainers, PBT, and visualization."""

    def __init__(self, run_dir: str | Path, filename: str = "metrics.jsonl"):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / filename

    def log(self, step: int, metrics: dict[str, Any]) -> None:
        row = {"time": time.time(), "step": int(step)}
        row.update(_jsonable(metrics))
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    def read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]


class EpisodeRecorder:
    """Stores selected real and imagined frames as compressed NumPy arrays."""

    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.frames_dir = self.run_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self._episode_frames: list[np.ndarray] = []

    def add_frame(self, grid: np.ndarray) -> None:
        self._episode_frames.append(np.asarray(grid, dtype=np.int8).copy())

    def flush_episode(self, episode_index: int) -> Path | None:
        if not self._episode_frames:
            return None
        path = self.frames_dir / f"episode_{episode_index:04d}.npz"
        np.savez_compressed(path, frames=np.stack(self._episode_frames))
        self._episode_frames.clear()
        return path

    def save_imagination(self, step: int, imagined_grids: list[np.ndarray]) -> Path | None:
        if not imagined_grids:
            return None
        path = self.frames_dir / f"imagination_step_{int(step):07d}.npz"
        np.savez_compressed(path, frames=np.stack([np.asarray(g, dtype=np.int8) for g in imagined_grids]))
        return path


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value
