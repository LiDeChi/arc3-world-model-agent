from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Iterable

import numpy as np


@dataclass
class Transition:
    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int = 50_000, seed: int = 0):
        self.capacity = int(capacity)
        self.rng = np.random.default_rng(seed)
        self._data: list[Transition] = []
        self._pos = 0

    def __len__(self) -> int:
        return len(self._data)

    def add(self, transition: Transition) -> None:
        if len(self._data) < self.capacity:
            self._data.append(transition)
        else:
            self._data[self._pos] = transition
        self._pos = (self._pos + 1) % self.capacity

    def extend(self, transitions: Iterable[Transition]) -> None:
        for transition in transitions:
            self.add(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        if not self._data:
            raise ValueError("Cannot sample from an empty replay buffer")
        idx = self.rng.integers(0, len(self._data), size=int(batch_size))
        return [self._data[int(i)] for i in idx]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump({"capacity": self.capacity, "data": self._data, "pos": self._pos}, f)

    @classmethod
    def load(cls, path: str | Path) -> "ReplayBuffer":
        with Path(path).open("rb") as f:
            payload = pickle.load(f)
        buf = cls(payload["capacity"])
        buf._data = payload["data"]
        buf._pos = payload["pos"]
        return buf
