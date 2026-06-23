from __future__ import annotations

import importlib
from typing import Any

import numpy as np

from .actions import Action, ActionKind, action_to_index, index_to_action


def _action_name_matches(kind: ActionKind, name: str) -> bool:
    name = str(name).upper()
    if kind == ActionKind.RESET:
        return name == "RESET"
    if kind == ActionKind.UP:
        return name in {"ACTION1", "UP", "ACTION1_UP"}
    if kind == ActionKind.DOWN:
        return name in {"ACTION2", "DOWN", "ACTION2_DOWN"}
    if kind == ActionKind.LEFT:
        return name in {"ACTION3", "LEFT", "ACTION3_LEFT"}
    if kind == ActionKind.RIGHT:
        return name in {"ACTION4", "RIGHT", "ACTION4_RIGHT"}
    if kind == ActionKind.INTERACT:
        return name in {"ACTION5", "INTERACT", "ACTION5_INTERACT"}
    if kind == ActionKind.CLICK:
        return name in {"ACTION6", "CLICK", "ACTION6_CLICK"}
    if kind == ActionKind.UNDO:
        return name in {"ACTION7", "UNDO", "ACTION7_UNDO"}
    return False


class ArcEngineEnv:
    """Best-effort adapter around the public ARC-AGI-3 toolkit.

    The current docs expose `arc_agi.Arcade` and `arc_agi.OperationMode`.
    Some preview examples and third-party code used `arcengine`, so this
    adapter tries the official package first and then the legacy name. The rest
    of the repo should never import toolkit objects directly.
    """

    def __init__(self, game_id: str, seed: int = 0, mode: str = "OFFLINE", environments_dir: str | None = None):
        self.game_id = game_id
        self.seed = seed
        self.mode = mode
        self._arcengine = self._load_sdk()
        self._arcade = self._make_arcade(environments_dir=environments_dir)
        self._env = self._make_env(game_id)

    @classmethod
    def is_available(cls) -> bool:
        return importlib.util.find_spec("arc_agi") is not None or importlib.util.find_spec("arcengine") is not None

    def reset(self) -> dict:
        result = self._env.reset() if hasattr(self._env, "reset") else self._env.step(self._sdk_action(ActionKind.RESET))
        return self._normalize_obs(result)

    def step(self, action: Action | int) -> tuple[dict, float, bool, dict]:
        if isinstance(action, int):
            action = index_to_action(action)
        sdk_action = self._sdk_action(action)
        try:
            result = self._env.step(sdk_action, data=None, reasoning=None)
        except TypeError:
            result = self._env.step(sdk_action)
        if isinstance(result, tuple) and len(result) == 4:
            obs, reward, done, info = result
        else:
            obs = result
            reward = float(getattr(result, "reward", 0.0))
            done = bool(getattr(result, "done", False))
            info = getattr(result, "info", {}) or {}
        return self._normalize_obs(obs), float(reward), bool(done), dict(info)

    def _load_sdk(self):
        try:
            return importlib.import_module("arc_agi")
        except ImportError:
            return importlib.import_module("arcengine")

    def _make_arcade(self, environments_dir: str | None = None):
        Arcade = getattr(self._arcengine, "Arcade")
        OperationMode = getattr(self._arcengine, "OperationMode", None)
        kwargs = {}
        if OperationMode is not None:
            kwargs["operation_mode"] = getattr(OperationMode, self.mode.upper(), self.mode)
        if environments_dir is not None:
            kwargs["environments_dir"] = environments_dir
        try:
            return Arcade(**kwargs)
        except TypeError:
            return Arcade()

    def _make_env(self, game_id: str):
        make = getattr(self._arcade, "make")
        try:
            env = make(game_id, seed=self.seed, render_mode=None)
        except TypeError:
            try:
                env = make(game_id, seed=self.seed)
            except TypeError:
                env = make(game_id)
        if env is None:
            raise RuntimeError(f"Failed to create ARC environment: {game_id}")
        return env

    def _sdk_action(self, action: Action | ActionKind):
        kind = action.kind if isinstance(action, Action) else action
        # Prefer official GameAction names if present.
        GameAction = getattr(self._arcengine, "GameAction", None)
        if GameAction is None:
            if hasattr(self._env, "action_space"):
                for candidate in self._env.action_space:
                    name = getattr(candidate, "name", "")
                    if _action_name_matches(kind, name):
                        return candidate
            return int(kind)
        mapping = {
            ActionKind.RESET: "RESET",
            ActionKind.UP: "ACTION1",
            ActionKind.DOWN: "ACTION2",
            ActionKind.LEFT: "ACTION3",
            ActionKind.RIGHT: "ACTION4",
            ActionKind.INTERACT: "ACTION5",
            ActionKind.CLICK: "ACTION6",
            ActionKind.UNDO: "ACTION7",
        }
        name = mapping[kind]
        return getattr(GameAction, name, int(kind))

    def _normalize_obs(self, obs: Any) -> dict:
        grid = None
        for attr in ("grid", "state", "observation", "image"):
            if hasattr(obs, attr):
                grid = getattr(obs, attr)
                break
        if grid is None and isinstance(obs, dict):
            grid = obs.get("grid") or obs.get("state") or obs.get("observation") or obs.get("image")
        if grid is None:
            arr = np.asarray(obs)
        else:
            arr = np.asarray(grid)
        if arr.ndim == 3:
            arr = arr[..., 0]
        arr = arr.astype(np.int8, copy=False)
        return {
            "grid": arr,
            "score": float(getattr(obs, "score", 0.0) if not isinstance(obs, dict) else obs.get("score", 0.0)),
            "available_actions": list(range(8)),
            "step": int(getattr(obs, "step", 0) if not isinstance(obs, dict) else obs.get("step", 0)),
        }
