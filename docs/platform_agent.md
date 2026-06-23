# Autonomous Training Platform Agent

The platform agent is the training loop around the ARC player. It is allowed to use a local LLM during research/training, but the Kaggle submission path must remain offline and must not call the LLM.

## What it does

One cycle performs:

1. **Collect evidence** from local project docs, recent training metrics, and previous platform decisions.
2. **Ask local DeepSeek through LiteLLM** for one small experiment proposal.
3. **Sanitize the proposal** so only safe bounded config fields can change.
4. **Run training** with the new config.
5. **Evaluate and score** the result.
6. **Generate a report** and write `platform_agent_events.jsonl` plus `platform_agent_summary.json`.

Allowed self-adjusted fields:

- `train.learning_rate`
- `train.epsilon`
- `train.action_penalty`
- `model.imagination_horizon`
- `model.planning_rollouts`

## Local DeepSeek smoke test

This uses the local LiteLLM gateway and the working DeepSeek alias discovered on this machine:

```bash
python3 -m arcagent.platform.agent_cli \
  --deepseek-smoke \
  --model opencode-go/deepseek-v4-flash \
  --run-dir runs/platform_smoke
```

Expected result includes:

```json
{"ok": true, "message": "ARC3_DEEPSEEK_SMOKE"}
```

## Run one autonomous cycle

```bash
python3 -m arcagent.platform.agent_cli \
  --cycles 1 \
  --steps-per-cycle 80 \
  --model opencode-go/deepseek-v4-flash \
  --run-dir runs/platform_agent_deepseek_smoke
```

Artifacts:

- `runs/platform_agent_deepseek_smoke/platform_agent_events.jsonl`
- `runs/platform_agent_deepseek_smoke/platform_agent_summary.json`
- `runs/platform_agent_deepseek_smoke/cycle_00/report.html`
- `runs/platform_agent_deepseek_smoke/cycle_00/checkpoints/agent.pkl`

## DeepSeek fallback behavior

The local LiteLLM config has direct aliases `deepseek-v4-flash` and `deepseek-v4-pro`, but during verification their upstream key returned an authentication error. The working DeepSeek route was:

- `opencode-go/deepseek-v4-flash`

The client keeps a fallback list, but this machine should prefer the `opencode-go/...` alias unless the direct DeepSeek key is repaired.

## Submission safety

The platform agent lives under `arcagent/platform/` and is training-only. The submission-critical path remains:

- `submission/agent.py`
- `arcagent/agent/`
- `arcagent/models/`
- `arcagent/envs/`

Run:

```bash
python3 -m submission.offline_check
```

before packaging a competition submission.
