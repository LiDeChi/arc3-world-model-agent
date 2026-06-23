from __future__ import annotations

import argparse

from arcagent.config import RunConfig
from arcagent.train import Trainer


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train the ARC Dreamer-lite agent")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--run-dir", type=str, default="runs/default")
    p.add_argument("--grid-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epsilon", type=float, default=0.15)
    p.add_argument("--learning-rate", type=float, default=3e-3)
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg = RunConfig(run_dir=args.run_dir)
    cfg.env.grid_size = args.grid_size
    cfg.env.seed = args.seed
    cfg.train.seed = args.seed
    cfg.train.steps = args.steps
    cfg.train.epsilon = args.epsilon
    cfg.train.learning_rate = args.learning_rate
    result = Trainer(cfg).train()
    print(f"trained steps={result.steps} episodes={result.episodes} best_return={result.best_return:.3f} run_dir={result.run_dir}")


if __name__ == "__main__":
    main()
