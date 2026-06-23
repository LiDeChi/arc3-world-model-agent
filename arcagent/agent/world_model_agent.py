from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np

from arcagent.envs.actions import legal_action_indices
from arcagent.models import DreamerLiteConfig, DreamerLiteWorldModel


@dataclass
class AgentConfig:
    grid_size: int = 16
    epsilon: float = 0.15
    gamma: float = 0.97
    imagination_horizon: int = 8
    planning_rollouts: int = 8
    action_penalty: float = 0.01
    seed: int = 0


class WorldModelAgent:
    """Prediction-first ARC agent.

    For every legal first action, it samples short imagined continuations from
    the world model and scores them by predicted reward minus action cost. This
    is the minimal runnable form of "connect multi-step consequences before
    acting".
    """

    def __init__(self, world_model: DreamerLiteWorldModel, config: AgentConfig):
        self.world_model = world_model
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.legal_actions = legal_action_indices(include_reset=False, include_undo=False, include_click=False)

    @classmethod
    def create(cls, grid_size: int = 16, latent_size: int = 64, seed: int = 0, **kwargs) -> "WorldModelAgent":
        wm = DreamerLiteWorldModel(DreamerLiteConfig(grid_size=grid_size, latent_size=latent_size, seed=seed))
        cfg = AgentConfig(grid_size=grid_size, seed=seed, **kwargs)
        return cls(wm, cfg)

    def act(self, obs: dict, explore: bool = True) -> int:
        available = [a for a in obs.get("available_actions", self.legal_actions) if a in self.legal_actions]
        if not available:
            available = self.legal_actions
        if explore and self.rng.random() < self.config.epsilon:
            return int(self.rng.choice(available))
        scores = {a: self._score_first_action(obs["grid"], a, available) for a in available}
        return int(max(scores, key=scores.get))

    def _score_first_action(self, grid: np.ndarray, first_action: int, available: list[int]) -> float:
        scores = []
        for _ in range(max(1, self.config.planning_rollouts)):
            actions = [first_action]
            for _h in range(self.config.imagination_horizon - 1):
                actions.append(int(self.rng.choice(available)))
            _grids, rewards = self.world_model.imagine(grid, actions)
            total = 0.0
            discount = 1.0
            for r in rewards:
                total += discount * (float(r) - self.config.action_penalty)
                discount *= self.config.gamma
            scores.append(total)
        return float(np.mean(scores))

    def train_world_model(self, batch, learning_rate: float) -> dict[str, float]:
        return self.world_model.train_batch(batch, learning_rate=learning_rate)

    def imagined_rollout(self, grid: np.ndarray, steps: int | None = None) -> list[np.ndarray]:
        steps = steps or self.config.imagination_horizon
        actions = [int(self.rng.choice(self.legal_actions)) for _ in range(steps)]
        grids, _rewards = self.world_model.imagine(grid, actions)
        return grids

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump({"config": self.config, "world_model": self.world_model.__dict__}, f)

    @classmethod
    def load(cls, path: str | Path) -> "WorldModelAgent":
        with Path(path).open("rb") as f:
            payload = pickle.load(f)
        wm_state = payload["world_model"]
        wm = DreamerLiteWorldModel(wm_state["config"])
        wm.__dict__.update(wm_state)
        return cls(wm, payload["config"])
