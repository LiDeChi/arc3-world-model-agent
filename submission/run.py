from __future__ import annotations

import argparse

from arcagent.envs import MockArcEnv
from submission.agent import SubmissionAgent


def main() -> None:
    p = argparse.ArgumentParser(description="Run local submission smoke test")
    p.add_argument("--checkpoint", default="runs/default/checkpoints/agent.pkl")
    p.add_argument("--episodes", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=80)
    args = p.parse_args()

    env = MockArcEnv(max_steps=args.max_steps)
    agent = SubmissionAgent(args.checkpoint)
    for ep in range(args.episodes):
        obs = env.reset()
        total = 0.0
        for step in range(args.max_steps):
            action_idx = agent.choose_action_index(obs)
            obs, reward, done, info = env.step(action_idx)
            total += float(reward)
            if done:
                break
        print(f"episode={ep} steps={step + 1} return={total:.3f} success={bool(info.get('success'))}")


if __name__ == "__main__":
    main()
