# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project overview

A lightweight, offline-safe research scaffold for ARC-AGI-3. It trains a NumPy-only "Dreamer-lite" world-model agent that predicts future grids and plans over imagined rollouts, plus a metrics-driven self-tuning training platform (PBT-style) and an autonomous LLM-driven platform agent for research/training only.

The inference-critical path for a Kaggle-style submission is intentionally small and offline-only:
- `submission/agent.py`
- `arcagent/agent/world_model_agent.py`
- `arcagent/models/dreamer_lite.py`
- `arcagent/envs/` (action mapping and fallback env)

## Common commands

All development assumes the local virtual environment at `.venv`:

```bash
source .venv/bin/activate
```

Install the package in editable mode:

```bash
python -m pip install -e .
```

Run the full test suite:

```bash
python -m unittest discover -s tests
```

Run a single test file:

```bash
python -m unittest tests.test_env
python -m unittest tests.test_model_agent
python -m unittest tests.test_platform_agent
python -m unittest tests.test_training_platform_submission
```

Run a single test method:

```bash
python -m unittest tests.test_training_platform_submission.IntegrationTests.test_training_report_submission_and_pbt
```

Train a single agent (smoke run):

```bash
python -m arcagent.train.run --steps 300 --run-dir runs/smoke
```

Generate an HTML report from a run directory:

```bash
python -m arcagent.viz.report runs/smoke
```

**Learn the codebase** using the interactive notebook series (the primary way this project is meant to be understood):

```bash
marimo edit notebooks/01_intro_arc3.ipynb     # or jupyter lab notebooks/
```

See `notebooks/README.md` for the full learning path.

Run the self-tuning PBT platform:

```bash
python -m arcagent.platform.pbt --generations 2 --population 4 --steps-per-generation 300 --run-dir runs/pbt
```

Run the fixed-seed smoke benchmark:

```bash
python -m arcagent.eval.benchmark --run-dir runs/benchmark
```

Run the autonomous LLM platform agent with local DeepSeek (training-time only):

```bash
python3 -m arcagent.platform.agent_cli \
  --cycles 1 \
  --steps-per-cycle 80 \
  --model opencode-go/deepseek-v4-flash \
  --run-dir runs/platform_agent_deepseek_smoke
```

Run the local submission smoke test:

```bash
python -m submission.run --episodes 2 --checkpoint runs/smoke/checkpoints/agent.pkl
```

Run the offline compliance check before packaging a submission:

```bash
python -m submission.offline_check
```

## Architecture

### Configuration

`arcagent.config` defines `RunConfig`, a nested dataclass of `EnvConfig`, `ModelConfig`, `TrainConfig`, and `PlatformConfig`. Configs are saved as JSON in every run directory (`config.json`) and can be loaded back with `RunConfig.load_json`. Use `with_overrides(config, train.learning_rate=0.001)` for dotted-path overrides.

### Environment layer

`arcagent.envs.arcengine_env.ArcEngineEnv` is a best-effort adapter around the official `arc_agi` toolkit (with a fallback to the legacy `arcengine` import name). For local smoke tests and most training, `arcagent.envs.mock_arc.MockArcEnv` is used automatically when the official toolkit is unavailable. Observations are dicts with `grid` (int8 NumPy array) and `available_actions`. All action encoding/decoding lives in `arcagent.envs.actions`.

### World model

`arcagent.models.dreamer_lite.DreamerLiteWorldModel` is a compact NumPy implementation of the predictive core: encode grid → latent, predict next latent conditioned on action, decode grid, predict reward. It is trained with manual SGD in `train_batch` on `(obs, action, reward, next_obs, done)` transitions. It is intentionally inspectable in notebooks, not a full RSSM.

### Agent

`arcagent.agent.world_model_agent.WorldModelAgent` wraps the world model. At decision time, for each legal first action it samples imagined continuations of length `imagination_horizon` and scores them by discounted predicted reward minus `action_penalty`. `act(..., explore=True)` adds epsilon-greedy exploration. `SubmissionAgent` loads a checkpointed `WorldModelAgent` and calls `act(..., explore=False)` for inference.

### Training

`arcagent.train.trainer.Trainer` owns the environment, agent, replay buffer, metric logger, and episode recorder. It runs a fixed number of `steps`, saving checkpoints to `{run_dir}/checkpoints/agent.pkl` and metrics to `{run_dir}/metrics.jsonl`.

### Self-tuning platform

`arcagent.platform.controller.PBTController` runs a population of trials each generation. Trials are ranked by `eval_return_mean + 2 * eval_success_rate`; weaker configs are replaced by elite configs and perturbed across a bounded set of hyperparameters (`train.learning_rate`, `train.epsilon`, `train.action_penalty`, `model.imagination_horizon`, `model.planning_rollouts`). Events are logged to `pbt_events.jsonl` and a summary to `pbt_summary.json`.

### Autonomous platform agent

`arcagent.platform.agent.PlatformAgent` uses a local LLM (via LiteLLM, defaulting to DeepSeek aliases) to propose one small experiment per cycle. It reads local docs, recent metrics, and prior decisions; sanitizes the proposal to the same bounded config fields as PBT; runs training; scores the result; and writes `platform_agent_events.jsonl` plus `platform_agent_summary.json`. This is training-time only and must not leak into the offline submission bundle.

### Submission

`submission/agent.SubmissionAgent` is the Kaggle-style inference wrapper. It loads a local checkpoint if present; otherwise it falls back to an untrained deterministic model so the entry point stays executable. `submission/offline_check.py` scans the inference-critical path for network/LLM/client patterns and must pass before any competition packaging.

### Reporting

`arcagent.viz.report.build_report` reads `metrics.jsonl` and `pbt_events.jsonl` from a run directory and writes a self-contained HTML report with episode returns, world-model loss sparklines, and a beginner explanation.

## Runtime dependencies

Core runtime requires only `numpy>=1.26`. Optional extras:
- `viz`: matplotlib, pandas, plotly
- `notebook`: jupyter, nbformat
- `dev`: pytest

## Important rules

- Keep the submission path (`submission/agent.py` and its dependencies in `arcagent/agent/`, `arcagent/models/`, `arcagent/envs/`) free of network calls, LLM clients, and external services.
- Run `python -m submission.offline_check` before considering a change ready for competition packaging.
- The LLM-driven platform agent (`arcagent/platform/`) is for training/research only; do not move it into the submission package.
