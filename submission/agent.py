from __future__ import annotations

from pathlib import Path

from arcagent.agent import WorldModelAgent
from arcagent.envs.actions import Action, index_to_action


class SubmissionAgent:
    """Tiny Kaggle-style inference wrapper.

    It loads a local checkpoint if available. If no checkpoint exists, it uses a
    deterministic untrained world model so the interface remains executable.
    """

    def __init__(self, checkpoint: str | Path | None = None, grid_size: int = 16):
        if checkpoint and Path(checkpoint).exists():
            self.agent = WorldModelAgent.load(checkpoint)
        else:
            self.agent = WorldModelAgent.create(grid_size=grid_size, seed=0, epsilon=0.0)

    def choose_action(self, observation: dict) -> Action:
        action_index = self.agent.act(observation, explore=False)
        grid_size = int(observation["grid"].shape[0])
        return index_to_action(action_index, grid_size=grid_size)

    def choose_action_index(self, observation: dict) -> int:
        return int(self.agent.act(observation, explore=False))
