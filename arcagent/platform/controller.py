from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import copy
import json

import numpy as np

from arcagent.config import RunConfig
from arcagent.data import MetricLogger
from arcagent.train import Trainer


@dataclass
class TrialState:
    trial_id: int
    generation: int
    run_dir: Path
    score: float
    config: RunConfig


class PBTController:
    """Metrics-driven self-tuning training platform.

    Each generation trains a small population, ranks trials by evaluation score,
    copies the best configuration into weaker trials, and perturbs a narrow set
    of stable hyperparameters. Every decision is logged for later visualization.
    """

    def __init__(self, base_config: RunConfig):
        self.base_config = base_config
        self.run_dir = Path(base_config.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.logger = MetricLogger(self.run_dir, filename="pbt_events.jsonl")
        self.rng = np.random.default_rng(base_config.platform.seed)

    def run(self) -> list[TrialState]:
        population = self._initial_population()
        states: list[TrialState] = []
        for generation in range(self.base_config.platform.generations):
            generation_states = []
            for trial_id, cfg in enumerate(population):
                cfg = copy.deepcopy(cfg)
                cfg.train.steps = self.base_config.platform.steps_per_generation
                cfg.run_dir = str(self.run_dir / f"gen_{generation:02d}" / f"trial_{trial_id:02d}")
                trainer = Trainer(cfg)
                result = trainer.train()
                eval_metrics = trainer.evaluate(episodes=2)
                score = float(eval_metrics["eval_return_mean"] + 2.0 * eval_metrics["eval_success_rate"])
                state = TrialState(trial_id, generation, Path(cfg.run_dir), score, cfg)
                generation_states.append(state)
                self.logger.log(generation, {"event": "trial_finished", "trial_id": trial_id, "score": score, "eval": eval_metrics, "config": cfg.to_dict()})
            states.extend(generation_states)
            population = self._exploit_explore(generation_states)
        self._write_summary(states)
        return states

    def _initial_population(self) -> list[RunConfig]:
        configs = []
        for i in range(self.base_config.platform.population):
            cfg = copy.deepcopy(self.base_config)
            factor = float(self.rng.choice([0.5, 0.8, 1.0, 1.2, 1.8]))
            cfg.train.learning_rate *= factor
            cfg.train.epsilon = float(np.clip(cfg.train.epsilon * self.rng.uniform(0.7, 1.5), 0.02, 0.6))
            cfg.train.action_penalty = float(np.clip(cfg.train.action_penalty * self.rng.uniform(0.5, 1.5), 0.001, 0.05))
            cfg.model.imagination_horizon = int(self.rng.choice([4, 6, 8, 10]))
            cfg.model.planning_rollouts = int(self.rng.choice([4, 8, 12]))
            cfg.train.seed = self.base_config.train.seed + i
            cfg.env.seed = self.base_config.env.seed + i
            configs.append(cfg)
        return configs

    def _exploit_explore(self, states: list[TrialState]) -> list[RunConfig]:
        ranked = sorted(states, key=lambda s: s.score, reverse=True)
        n = len(ranked)
        elite_count = max(1, int(round(n * self.base_config.platform.exploit_fraction)))
        elites = ranked[:elite_count]
        next_population: list[RunConfig] = []
        for rank, state in enumerate(ranked):
            if rank < elite_count:
                new_cfg = copy.deepcopy(state.config)
                decision = "keep_elite"
                source = state.trial_id
            else:
                donor = elites[rank % elite_count]
                new_cfg = copy.deepcopy(donor.config)
                decision = "copy_and_perturb"
                source = donor.trial_id
                self._perturb(new_cfg)
            self.logger.log(
                state.generation,
                {"event": "pbt_decision", "target_trial": state.trial_id, "decision": decision, "source_trial": source, "new_config": new_cfg.to_dict()},
            )
            next_population.append(new_cfg)
        return next_population

    def _perturb(self, cfg: RunConfig) -> None:
        lo, hi = self.base_config.platform.perturb_factors
        cfg.train.learning_rate = float(np.clip(cfg.train.learning_rate * self.rng.choice([lo, hi]), 1e-4, 2e-2))
        cfg.train.epsilon = float(np.clip(cfg.train.epsilon * self.rng.choice([lo, hi]), 0.02, 0.6))
        cfg.train.action_penalty = float(np.clip(cfg.train.action_penalty * self.rng.choice([lo, hi]), 0.001, 0.05))
        cfg.model.imagination_horizon = int(np.clip(cfg.model.imagination_horizon + int(self.rng.choice([-2, 0, 2])), 3, 15))
        cfg.model.planning_rollouts = int(np.clip(cfg.model.planning_rollouts + int(self.rng.choice([-4, 0, 4])), 2, 24))
        cfg.train.seed += 100
        cfg.env.seed += 100

    def _write_summary(self, states: list[TrialState]) -> None:
        best = max(states, key=lambda s: s.score) if states else None
        summary = {
            "best_score": None if best is None else best.score,
            "best_run_dir": None if best is None else str(best.run_dir),
            "trials": [
                {"trial_id": s.trial_id, "generation": s.generation, "score": s.score, "run_dir": str(s.run_dir)} for s in states
            ],
        }
        (self.run_dir / "pbt_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
