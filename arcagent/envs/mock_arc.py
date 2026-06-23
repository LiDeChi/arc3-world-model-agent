from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .actions import Action, ActionKind, action_to_index, index_to_action, legal_action_indices


@dataclass
class MockObservation:
    grid: np.ndarray
    score: float
    available_actions: list[int]
    step: int


class MockArcEnv:
    """Small deterministic ARC-like task for local training and tests.

    The grid uses ARC's 0-15 color vocabulary. An agent (color 2) must navigate
    around walls (color 8), pick up a key (color 4) using INTERACT, then reach
    the goal (color 3). This is intentionally tiny but requires multi-step
    dependency: goal reward is available only after the key is collected.
    """

    def __init__(self, grid_size: int = 16, max_steps: int = 80, seed: int = 0):
        self.grid_size = int(grid_size)
        self.max_steps = int(max_steps)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.history: list[tuple[tuple[int, int], bool, float]] = []
        self.reset()

    def reset(self) -> dict:
        self.steps = 0
        self.score = 0.0
        self.agent = (1, 1)
        self.has_key = False
        self.done = False
        self.key = (self.grid_size // 2, max(2, self.grid_size // 3))
        self.goal = (self.grid_size - 2, self.grid_size - 2)
        self.walls = self._make_walls()
        self.history.clear()
        return self._obs()

    def _make_walls(self) -> set[tuple[int, int]]:
        walls: set[tuple[int, int]] = set()
        mid = self.grid_size // 2
        for y in range(2, self.grid_size - 2):
            if y not in (self.grid_size // 3, self.grid_size - 3):
                walls.add((mid, y))
        for x in range(3, self.grid_size - 3):
            if x != mid - 1:
                walls.add((x, self.grid_size // 2 + 1))
        return walls

    def _grid(self) -> np.ndarray:
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        grid[0, :] = 8
        grid[-1, :] = 8
        grid[:, 0] = 8
        grid[:, -1] = 8
        for x, y in self.walls:
            grid[y, x] = 8
        grid[self.goal[1], self.goal[0]] = 3
        if not self.has_key:
            grid[self.key[1], self.key[0]] = 4
        grid[self.agent[1], self.agent[0]] = 2 if not self.has_key else 6
        return grid

    def _obs(self) -> dict:
        return {
            "grid": self._grid(),
            "score": float(self.score),
            "available_actions": legal_action_indices(include_reset=True, include_undo=True),
            "step": self.steps,
        }

    def step(self, action: Action | int) -> tuple[dict, float, bool, dict]:
        if isinstance(action, int):
            action = index_to_action(action, grid_size=self.grid_size)
        if self.done:
            return self._obs(), 0.0, True, {"reason": "already_done"}

        self.history.append((self.agent, self.has_key, self.score))
        self.steps += 1
        reward = -0.01
        info = {"success": False}

        if action.kind == ActionKind.RESET:
            obs = self.reset()
            return obs, -0.05, False, {"reset": True}
        if action.kind == ActionKind.UNDO and self.history:
            self.agent, self.has_key, self.score = self.history.pop()
            return self._obs(), -0.02, False, {"undo": True}

        dx, dy = 0, 0
        if action.kind == ActionKind.UP:
            dy = -1
        elif action.kind == ActionKind.DOWN:
            dy = 1
        elif action.kind == ActionKind.LEFT:
            dx = -1
        elif action.kind == ActionKind.RIGHT:
            dx = 1
        elif action.kind == ActionKind.INTERACT:
            if self._manhattan(self.agent, self.key) <= 1 and not self.has_key:
                self.has_key = True
                reward += 0.75
                info["picked_key"] = True
            else:
                reward -= 0.03
        elif action.kind == ActionKind.CLICK:
            # In real ARC-AGI-3 click is useful in some games. Here it marks intent
            # and gives a tiny bonus if clicked near the goal after collecting key.
            if self.has_key and action.x is not None and action.y is not None:
                if self._manhattan((action.x, action.y), self.goal) <= 1:
                    reward += 0.05

        if dx or dy:
            nx, ny = self.agent[0] + dx, self.agent[1] + dy
            blocked = (nx, ny) in self.walls or nx <= 0 or ny <= 0 or nx >= self.grid_size - 1 or ny >= self.grid_size - 1
            if blocked:
                reward -= 0.08
                info["blocked"] = True
            else:
                old_dist = self._goal_distance()
                self.agent = (nx, ny)
                if self._goal_distance() < old_dist:
                    reward += 0.015

        if self.agent == self.goal and self.has_key:
            reward += 2.0
            self.score = 1.0
            self.done = True
            info["success"] = True
        elif self.steps >= self.max_steps:
            self.done = True
            info["timeout"] = True

        return self._obs(), float(reward), bool(self.done), info

    def render(self) -> np.ndarray:
        return self._grid()

    def clone_state(self) -> dict:
        return {
            "steps": self.steps,
            "score": self.score,
            "agent": self.agent,
            "has_key": self.has_key,
            "done": self.done,
            "history": list(self.history),
        }

    def restore_state(self, state: dict) -> None:
        self.steps = state["steps"]
        self.score = state["score"]
        self.agent = tuple(state["agent"])
        self.has_key = bool(state["has_key"])
        self.done = bool(state["done"])
        self.history = list(state["history"])

    def _goal_distance(self) -> int:
        target = self.key if not self.has_key else self.goal
        return self._manhattan(self.agent, target)

    @staticmethod
    def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
