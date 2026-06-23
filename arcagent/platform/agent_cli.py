from __future__ import annotations

import argparse
from dataclasses import asdict
import json

from arcagent.config import RunConfig
from arcagent.platform.agent import DEFAULT_DEEPSEEK_MODELS, PlatformAgent


def main() -> None:
    p = argparse.ArgumentParser(description="Autonomous ARC3 training-platform agent")
    p.add_argument("--run-dir", default="runs/platform_agent")
    p.add_argument("--cycles", type=int, default=1)
    p.add_argument("--steps-per-cycle", type=int, default=120)
    p.add_argument("--model", action="append", dest="models", help="LiteLLM model alias; can be repeated")
    p.add_argument("--deepseek-smoke", action="store_true", help="Only test local LiteLLM DeepSeek connectivity")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cfg = RunConfig(run_dir=args.run_dir)
    cfg.train.seed = args.seed
    cfg.env.seed = args.seed
    agent = PlatformAgent(cfg, models=args.models or DEFAULT_DEEPSEEK_MODELS)

    if args.deepseek_smoke:
        result = agent.smoke_deepseek()
        print(json.dumps({"ok": result.ok, "model": result.model, "content": result.content, "error": result.error}, ensure_ascii=False, indent=2))
        raise SystemExit(0 if result.ok else 1)

    outcomes = agent.run(cycles=args.cycles, steps_per_cycle=args.steps_per_cycle)
    print(json.dumps({"outcomes": [asdict(o) for o in outcomes]}, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
