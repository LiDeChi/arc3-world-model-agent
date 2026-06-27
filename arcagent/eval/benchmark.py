from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from arcagent.agent import WorldModelAgent
from arcagent.agent.world_model_agent import AgentConfig
from arcagent.config import RunConfig
from arcagent.envs import MockArcEnv
from arcagent.models import DreamerLiteConfig, DreamerLiteWorldModel
from arcagent.platform import PBTController
from arcagent.train import Trainer


def evaluate_agent(agent: WorldModelAgent, config: RunConfig, episodes: int) -> dict[str, float]:
    """Evaluate an agent on fresh deterministic MockArcEnv episodes."""
    returns: list[float] = []
    successes = 0
    for episode in range(episodes):
        env = MockArcEnv(grid_size=config.env.grid_size, max_steps=config.env.max_steps, seed=config.env.seed + episode)
        obs = env.reset()
        total = 0.0
        for _step in range(config.env.max_steps):
            action = agent.act(obs, explore=False)
            obs, reward, done, info = env.step(action)
            total += float(reward)
            if done:
                successes += int(bool(info.get("success")))
                break
        returns.append(total)
    return {
        "eval_return_mean": float(np.mean(returns)) if returns else 0.0,
        "eval_success_rate": float(successes / max(1, episodes)),
    }


def _base_config(run_dir: Path, seed: int, log_every: int) -> RunConfig:
    cfg = RunConfig(run_dir=str(run_dir))
    cfg.env.seed = seed
    cfg.train.seed = seed
    cfg.platform.seed = seed
    cfg.train.batch_size = 8
    cfg.train.log_every = log_every
    cfg.train.eval_every = max(50, log_every * 4)
    return cfg


def _untrained_agent(config: RunConfig) -> WorldModelAgent:
    world_model = DreamerLiteWorldModel(
        DreamerLiteConfig(
            grid_size=config.env.grid_size,
            latent_size=config.model.latent_size,
            learning_rate=config.train.learning_rate,
            seed=config.train.seed,
        )
    )
    agent_config = AgentConfig(
        grid_size=config.env.grid_size,
        epsilon=config.train.epsilon,
        gamma=config.train.gamma,
        imagination_horizon=config.model.imagination_horizon,
        planning_rollouts=config.model.planning_rollouts,
        action_penalty=config.train.action_penalty,
        seed=config.train.seed,
    )
    return WorldModelAgent(world_model, agent_config)


def _row(name: str, metrics: dict[str, float], **extra: Any) -> dict[str, Any]:
    return {"name": name, **metrics, **extra}


def run_benchmark(
    run_dir: str | Path = "runs/benchmark",
    *,
    seed: int = 0,
    episodes: int = 3,
    train_steps: int = 300,
    pbt_generations: int = 1,
    pbt_population: int = 2,
    pbt_steps_per_generation: int = 150,
    log_every: int = 25,
) -> dict[str, Any]:
    """Run a small fixed-seed benchmark across baseline, training, and PBT."""
    out_dir = Path(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_cfg = _base_config(out_dir / "untrained", seed, log_every)
    baseline_cfg.save_json(Path(baseline_cfg.run_dir) / "config.json")
    baseline_agent = _untrained_agent(baseline_cfg)
    rows = [
        _row(
            "untrained",
            evaluate_agent(baseline_agent, baseline_cfg, episodes),
            run_dir=str(Path(baseline_cfg.run_dir)),
        )
    ]

    single_cfg = _base_config(out_dir / "single_train", seed, log_every)
    single_cfg.train.steps = train_steps
    single_trainer = Trainer(single_cfg)
    single_result = single_trainer.train()
    rows.append(
        _row(
            "single_train",
            single_trainer.evaluate(episodes=episodes),
            train_steps=train_steps,
            run_dir=str(single_result.run_dir),
        )
    )

    pbt_cfg = _base_config(out_dir / "tiny_pbt", seed, log_every)
    pbt_cfg.platform.generations = pbt_generations
    pbt_cfg.platform.population = pbt_population
    pbt_cfg.platform.steps_per_generation = pbt_steps_per_generation
    pbt_states = PBTController(pbt_cfg).run()
    best_state = max(pbt_states, key=lambda state: state.score)
    best_agent = WorldModelAgent.load(best_state.run_dir / "checkpoints" / "agent.pkl")
    rows.append(
        _row(
            "tiny_pbt",
            evaluate_agent(best_agent, best_state.config, episodes),
            best_score=float(best_state.score),
            generations=pbt_generations,
            population=pbt_population,
            steps_per_generation=pbt_steps_per_generation,
            run_dir=str(best_state.run_dir),
        )
    )

    result = {
        "seed": seed,
        "episodes": episodes,
        "benchmark": rows,
        "best_by_eval_return": max(rows, key=lambda row: row["eval_return_mean"])["name"],
    }
    (out_dir / "benchmark.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _print_table(result: dict[str, Any]) -> None:
    print("name             return_mean  success_rate  run_dir")
    for row in result["benchmark"]:
        print(
            f"{row['name']:<16} "
            f"{row['eval_return_mean']:>11.3f}  "
            f"{row['eval_success_rate']:>12.3f}  "
            f"{row['run_dir']}"
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a fixed-seed ARC3 smoke benchmark.")
    parser.add_argument("--run-dir", default="runs/benchmark")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--train-steps", type=int, default=300)
    parser.add_argument("--pbt-generations", type=int, default=1)
    parser.add_argument("--pbt-population", type=int, default=2)
    parser.add_argument("--pbt-steps-per-generation", type=int, default=150)
    parser.add_argument("--log-every", type=int, default=25)
    args = parser.parse_args(argv)
    result = run_benchmark(
        args.run_dir,
        seed=args.seed,
        episodes=args.episodes,
        train_steps=args.train_steps,
        pbt_generations=args.pbt_generations,
        pbt_population=args.pbt_population,
        pbt_steps_per_generation=args.pbt_steps_per_generation,
        log_every=args.log_every,
    )
    _print_table(result)
    print(f"wrote {Path(args.run_dir) / 'benchmark.json'}")


if __name__ == "__main__":
    main()
