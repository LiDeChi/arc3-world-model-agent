# ARC-AGI-3 World-Model Agent + Self-Tuning Trainer

This repo is a compact, offline-safe ARC-AGI-3 research scaffold. It contains:

- a turn-based ARC-style environment wrapper that tries the official `arc_agi` toolkit and includes a deterministic `MockArcEnv` for local smoke tests;
- a NumPy **Dreamer-lite** world-model agent that learns to predict future grids, imagine rollouts, and plan over candidate actions;
- a metrics-driven self-tuning training platform inspired by Population-Based Training (PBT);
- full training logs, visual reports, and a complete **interactive notebook series** designed for learning the code step-by-step (see `notebooks/README.md`);
- a Kaggle-style offline submission shim with a static compliance check.

The executable v1 deliberately avoids heavyweight runtime dependencies so it runs on a small local machine. The architecture mirrors the competition design target: learn a predictive model of the grid world, connect multi-step consequences through imagined rollouts, and optimize action efficiency.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m arcagent.train.run --steps 300 --run-dir runs/smoke
python -m arcagent.viz.monitor runs/smoke
python -m arcagent.viz.report runs/smoke
python -m submission.offline_check
python -m unittest discover -s tests
```

The live monitor opens at `http://127.0.0.1:8765` and refreshes while training writes JSONL logs. The static report is written to `runs/smoke/report.html`.

## Learn the Project Interactively (Recommended)

The best way to understand this codebase is through the notebooks:

```bash
# Using marimo (you already installed it!)
marimo edit notebooks/01_intro_arc3.ipynb

# or traditional Jupyter
python -m pip install -e ".[notebook]"
jupyter lab notebooks/
```

See the full guided learning path: [notebooks/README.md](notebooks/README.md)

The series progressively covers:
Env → World Model (imagination) → Planning Agent → PBT self-tuning → Results → Submission

## Main commands

```bash
# Train one Dreamer-lite agent on the local mock ARC task.
python -m arcagent.train.run --steps 1000 --run-dir runs/agent

# Run the self-tuning platform with several trials.
python -m arcagent.platform.pbt --generations 2 --population 4 --steps-per-generation 300 --run-dir runs/pbt

# Compare untrained, single-train, and tiny-PBT agents with a fixed seed.
python -m arcagent.eval.benchmark --run-dir runs/benchmark

# Generate visual report from any run directory.
python -m arcagent.viz.report runs/pbt

# Watch a run while it is training.
python -m arcagent.viz.monitor runs/pbt --port 8765

# Exercise the submission shim locally.
python -m submission.run --episodes 2
```


## Autonomous platform agent with local DeepSeek

Run a simple LiteLLM/DeepSeek connectivity check:

```bash
python3 -m arcagent.platform.agent_cli --deepseek-smoke --model opencode-go/deepseek-v4-flash --run-dir runs/platform_smoke
```

Run one autonomous improvement cycle:

```bash
python3 -m arcagent.platform.agent_cli --cycles 1 --steps-per-cycle 80 --model opencode-go/deepseek-v4-flash --run-dir runs/platform_agent_deepseek_smoke
```

See `docs/platform_agent.md` for details. The platform agent is training-only and is excluded from the offline submission-critical path.

## Competition safety notes

The `submission/` package uses only local Python/NumPy code and local checkpoints. It does not call external services. The self-tuning controller, notebooks, and visualization code are training-time tools and should not be included in a minimal Kaggle inference bundle unless the competition packaging allows it.

## License

MIT-0. See `LICENSE`.
