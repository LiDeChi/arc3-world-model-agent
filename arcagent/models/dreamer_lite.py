from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np


@dataclass
class DreamerLiteConfig:
    grid_size: int = 16
    colors: int = 16
    latent_size: int = 64
    learning_rate: float = 3e-3
    seed: int = 0


class DreamerLiteWorldModel:
    """Small NumPy world model for local, dependency-light ARC research.

    It is intentionally simpler than production Dreamer/RSSM, but preserves the
    crucial idea: encode a grid, predict the next latent state conditioned on an
    action, decode the imagined next grid, and predict reward. This makes the
    imagination loop inspectable in notebooks.
    """

    def __init__(self, config: DreamerLiteConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        flat = config.grid_size * config.grid_size * config.colors
        scale = 1.0 / np.sqrt(max(flat, 1))
        self.W_enc = self.rng.normal(0, scale, size=(flat, config.latent_size)).astype(np.float32)
        self.W_dyn = self.rng.normal(0, 0.05, size=(config.latent_size + 8, config.latent_size)).astype(np.float32)
        self.b_dyn = np.zeros(config.latent_size, dtype=np.float32)
        self.W_dec = self.rng.normal(0, 0.05, size=(config.latent_size, flat)).astype(np.float32)
        self.b_dec = np.zeros(flat, dtype=np.float32)
        self.W_reward = self.rng.normal(0, 0.05, size=(config.latent_size + 8, 1)).astype(np.float32)
        self.b_reward = np.zeros(1, dtype=np.float32)

    def encode(self, grid: np.ndarray) -> np.ndarray:
        onehot = self._onehot(grid).reshape(-1).astype(np.float32)
        latent = np.tanh(onehot @ self.W_enc)
        return latent.astype(np.float32)

    def predict_next_latent(self, latent: np.ndarray, action: int) -> np.ndarray:
        inp = np.concatenate([latent.astype(np.float32), self._action_onehot(action)])
        return np.tanh(inp @ self.W_dyn + self.b_dyn).astype(np.float32)

    def predict_reward(self, latent: np.ndarray, action: int) -> float:
        inp = np.concatenate([latent.astype(np.float32), self._action_onehot(action)])
        return float((inp @ self.W_reward + self.b_reward)[0])

    def decode_logits(self, latent: np.ndarray) -> np.ndarray:
        logits = latent.astype(np.float32) @ self.W_dec + self.b_dec
        return logits.reshape(self.config.grid_size, self.config.grid_size, self.config.colors)

    def decode(self, latent: np.ndarray) -> np.ndarray:
        return np.argmax(self.decode_logits(latent), axis=-1).astype(np.int8)

    def imagine(self, grid: np.ndarray, actions: list[int]) -> tuple[list[np.ndarray], list[float]]:
        latent = self.encode(grid)
        grids: list[np.ndarray] = []
        rewards: list[float] = []
        for action in actions:
            rewards.append(self.predict_reward(latent, action))
            latent = self.predict_next_latent(latent, action)
            grids.append(self.decode(latent))
        return grids, rewards

    def train_batch(self, batch, learning_rate: float | None = None) -> dict[str, float]:
        lr = float(learning_rate or self.config.learning_rate)
        losses = []
        reward_losses = []
        dyn_losses = []
        # Manual SGD for a compact single-step predictive objective.
        for tr in batch:
            obs = np.asarray(tr.obs, dtype=np.int8)
            next_obs = np.asarray(tr.next_obs, dtype=np.int8)
            action = int(tr.action)
            latent = self.encode(obs)
            target_latent = self.encode(next_obs)
            inp = np.concatenate([latent, self._action_onehot(action)])
            pre_dyn = inp @ self.W_dyn + self.b_dyn
            pred_latent = np.tanh(pre_dyn)
            dyn_err = pred_latent - target_latent
            dyn_loss = float(np.mean(dyn_err ** 2))
            dpre = (2.0 / dyn_err.size) * dyn_err * (1.0 - pred_latent ** 2)
            self.W_dyn -= lr * np.outer(inp, dpre).astype(np.float32)
            self.b_dyn -= lr * dpre.astype(np.float32)

            logits = pred_latent @ self.W_dec + self.b_dec
            target = self._onehot(next_obs).reshape(-1).astype(np.float32)
            recon_err = logits - target
            recon_loss = float(np.mean(recon_err ** 2))
            dlogits = (2.0 / recon_err.size) * recon_err
            self.W_dec -= lr * np.outer(pred_latent, dlogits).astype(np.float32)
            self.b_dec -= lr * dlogits.astype(np.float32)

            pred_reward = float((inp @ self.W_reward + self.b_reward)[0])
            rerr = pred_reward - float(tr.reward)
            self.W_reward -= lr * (2.0 * rerr * inp[:, None]).astype(np.float32)
            self.b_reward -= lr * np.array([2.0 * rerr], dtype=np.float32)

            losses.append(recon_loss + dyn_loss + rerr * rerr)
            reward_losses.append(float(rerr * rerr))
            dyn_losses.append(dyn_loss)
        return {
            "world_model_loss": float(np.mean(losses)) if losses else 0.0,
            "dynamics_loss": float(np.mean(dyn_losses)) if dyn_losses else 0.0,
            "reward_loss": float(np.mean(reward_losses)) if reward_losses else 0.0,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, path: str | Path) -> "DreamerLiteWorldModel":
        with Path(path).open("rb") as f:
            state = pickle.load(f)
        obj = cls(state["config"])
        obj.__dict__.update(state)
        return obj

    def _onehot(self, grid: np.ndarray) -> np.ndarray:
        grid = np.asarray(grid, dtype=np.int64)
        grid = np.clip(grid, 0, self.config.colors - 1)
        return np.eye(self.config.colors, dtype=np.float32)[grid]

    @staticmethod
    def _action_onehot(action: int) -> np.ndarray:
        out = np.zeros(8, dtype=np.float32)
        out[int(action) % 8] = 1.0
        return out
