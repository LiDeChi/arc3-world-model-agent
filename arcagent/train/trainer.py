from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from arcagent.agent import WorldModelAgent
from arcagent.agent.world_model_agent import AgentConfig
from arcagent.config import RunConfig
from arcagent.data import EpisodeRecorder, MetricLogger, ReplayBuffer, Transition
from arcagent.envs import ArcEngineEnv, MockArcEnv
from arcagent.models import DreamerLiteConfig, DreamerLiteWorldModel


@dataclass
class TrainResult:
    run_dir: Path
    episodes: int
    steps: int
    best_return: float
    final_return: float


class Trainer:
    def __init__(self, config: RunConfig):
        self.config = config
        self.run_dir = Path(config.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.config.save_json(self.run_dir / "config.json")
        self.logger = MetricLogger(self.run_dir)
        self.recorder = EpisodeRecorder(self.run_dir)
        self.env = self._make_env()
        self.agent = self._make_agent()
        self.buffer = ReplayBuffer(capacity=50_000, seed=config.train.seed)

    def _make_env(self):
        if self.config.env.name in {"arc_agi", "arcengine"}:
            try:
                return ArcEngineEnv(self.config.env.game_id, seed=self.config.env.seed)
            except Exception as exc:
                self.logger.log(0, {"event": "arcengine_fallback", "error": repr(exc)})
        return MockArcEnv(grid_size=self.config.env.grid_size, max_steps=self.config.env.max_steps, seed=self.config.env.seed)

    def _make_agent(self) -> WorldModelAgent:
        wm = DreamerLiteWorldModel(
            DreamerLiteConfig(
                grid_size=self.config.env.grid_size,
                latent_size=self.config.model.latent_size,
                learning_rate=self.config.train.learning_rate,
                seed=self.config.train.seed,
            )
        )
        cfg = AgentConfig(
            grid_size=self.config.env.grid_size,
            epsilon=self.config.train.epsilon,
            gamma=self.config.train.gamma,
            imagination_horizon=self.config.model.imagination_horizon,
            planning_rollouts=self.config.model.planning_rollouts,
            action_penalty=self.config.train.action_penalty,
            seed=self.config.train.seed,
        )
        return WorldModelAgent(wm, cfg)

    def train(self) -> TrainResult:
        obs = self.env.reset()
        episode_return = 0.0
        episode = 0
        best_return = -1e9
        returns = []
        self.recorder.add_frame(obs["grid"])

        for step in range(1, self.config.train.steps + 1):
            action = self.agent.act(obs, explore=True)
            next_obs, reward, done, info = self.env.step(action)
            shaped_reward = float(reward) - self.config.train.action_penalty
            self.buffer.add(Transition(obs=obs["grid"], action=action, reward=shaped_reward, next_obs=next_obs["grid"], done=done))
            self.recorder.add_frame(next_obs["grid"])
            episode_return += float(reward)

            losses = {}
            if len(self.buffer) >= max(4, self.config.train.batch_size // 2):
                batch = self.buffer.sample(min(self.config.train.batch_size, len(self.buffer)))
                losses = self.agent.train_world_model(batch, learning_rate=self.config.train.learning_rate)

            if step % self.config.train.log_every == 0:
                self.logger.log(
                    step,
                    {
                        "episode": episode,
                        "buffer_size": len(self.buffer),
                        "episode_return": episode_return,
                        "epsilon": self.agent.config.epsilon,
                        **losses,
                    },
                )

            if step % max(50, self.config.train.eval_every // 2) == 0:
                imagined = self.agent.imagined_rollout(obs["grid"], steps=4)
                self.recorder.save_imagination(step, imagined)

            obs = next_obs
            if done:
                returns.append(episode_return)
                best_return = max(best_return, episode_return)
                self.logger.log(step, {"event": "episode_end", "episode": episode, "return": episode_return, "success": bool(info.get("success"))})
                self.recorder.flush_episode(episode)
                episode += 1
                episode_return = 0.0
                obs = self.env.reset()
                self.recorder.add_frame(obs["grid"])

        self.agent.save(self.run_dir / "checkpoints" / "agent.pkl")
        final_return = returns[-1] if returns else episode_return
        self.logger.log(self.config.train.steps, {"event": "training_end", "episodes": episode, "best_return": best_return, "final_return": final_return})
        return TrainResult(self.run_dir, episode, self.config.train.steps, float(best_return), float(final_return))

    def evaluate(self, episodes: int = 3) -> dict[str, float]:
        returns = []
        successes = 0
        for _ in range(episodes):
            obs = self.env.reset()
            total = 0.0
            for _step in range(self.config.env.max_steps):
                action = self.agent.act(obs, explore=False)
                obs, reward, done, info = self.env.step(action)
                total += float(reward)
                if done:
                    successes += int(bool(info.get("success")))
                    break
            returns.append(total)
        return {"eval_return_mean": float(np.mean(returns)), "eval_success_rate": successes / max(1, episodes)}
