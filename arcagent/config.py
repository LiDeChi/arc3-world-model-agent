from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
from typing import Any, Dict


@dataclass
class EnvConfig:
    name: str = "mock"
    game_id: str = "mock_navigation"
    grid_size: int = 16
    max_steps: int = 80
    seed: int = 0


@dataclass
class ModelConfig:
    latent_size: int = 64
    hidden_size: int = 128
    imagination_horizon: int = 8
    planning_candidates: int = 3
    planning_rollouts: int = 8
    reconstruction_weight: float = 1.0
    dynamics_weight: float = 0.5
    reward_weight: float = 1.0


@dataclass
class TrainConfig:
    steps: int = 1000
    batch_size: int = 32
    sequence_length: int = 6
    learning_rate: float = 3e-3
    epsilon: float = 0.15
    gamma: float = 0.97
    action_penalty: float = 0.01
    curiosity_weight: float = 0.02
    log_every: int = 25
    eval_every: int = 200
    seed: int = 0


@dataclass
class PlatformConfig:
    population: int = 4
    generations: int = 3
    steps_per_generation: int = 500
    exploit_fraction: float = 0.25
    perturb_factors: tuple[float, float] = (0.8, 1.2)
    seed: int = 0


@dataclass
class RunConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    platform: PlatformConfig = field(default_factory=PlatformConfig)
    run_dir: str = "runs/default"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunConfig":
        return cls(
            env=EnvConfig(**data.get("env", {})),
            model=ModelConfig(**data.get("model", {})),
            train=TrainConfig(**data.get("train", {})),
            platform=PlatformConfig(**data.get("platform", {})),
            run_dir=data.get("run_dir", "runs/default"),
        )

    @classmethod
    def load_json(cls, path: str | Path) -> "RunConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def with_overrides(config: RunConfig, **kwargs: Any) -> RunConfig:
    """Return a copy with dotted overrides, e.g. train.learning_rate=0.001."""
    data = config.to_dict()
    for key, value in kwargs.items():
        parts = key.split(".")
        cursor = data
        for part in parts[:-1]:
            cursor = cursor[part]
        cursor[parts[-1]] = value
    return RunConfig.from_dict(data)
