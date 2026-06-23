from __future__ import annotations

import argparse

from arcagent.config import RunConfig
from arcagent.platform import PBTController


def main() -> None:
    p = argparse.ArgumentParser(description="Run the self-tuning PBT platform")
    p.add_argument("--generations", type=int, default=2)
    p.add_argument("--population", type=int, default=4)
    p.add_argument("--steps-per-generation", type=int, default=300)
    p.add_argument("--run-dir", type=str, default="runs/pbt")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cfg = RunConfig(run_dir=args.run_dir)
    cfg.platform.generations = args.generations
    cfg.platform.population = args.population
    cfg.platform.steps_per_generation = args.steps_per_generation
    cfg.platform.seed = args.seed
    cfg.train.seed = args.seed
    cfg.env.seed = args.seed
    states = PBTController(cfg).run()
    best = max(states, key=lambda s: s.score)
    print(f"pbt_complete trials={len(states)} best_score={best.score:.3f} best_run_dir={best.run_dir}")


if __name__ == "__main__":
    main()
