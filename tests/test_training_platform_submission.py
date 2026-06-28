import tempfile
import unittest
from pathlib import Path

import numpy as np

from arcagent.config import RunConfig
from arcagent.platform import PBTController
from arcagent.train import Trainer
from arcagent.viz import build_report, collect_run_state
from submission.offline_check import scan
from submission.agent import SubmissionAgent


class IntegrationTests(unittest.TestCase):
    def test_training_report_submission_and_pbt(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = RunConfig(run_dir=str(Path(tmp) / "train"))
            cfg.train.steps = 40
            cfg.train.batch_size = 4
            cfg.train.log_every = 10
            result = Trainer(cfg).train()
            self.assertTrue((result.run_dir / "checkpoints" / "agent.pkl").exists())
            report = build_report(result.run_dir)
            self.assertTrue(report.exists())
            sub = SubmissionAgent(result.run_dir / "checkpoints" / "agent.pkl")
            obs = {"grid": __import__("numpy").zeros((16, 16), dtype="int8"), "available_actions": [1, 2, 3, 4, 5]}
            self.assertIn(sub.choose_action_index(obs), [1, 2, 3, 4, 5])
            self.assertEqual(scan(Path(__file__).resolve().parents[1]), [])

            pbt_cfg = RunConfig(run_dir=str(Path(tmp) / "pbt"))
            pbt_cfg.platform.population = 2
            pbt_cfg.platform.generations = 1
            pbt_cfg.platform.steps_per_generation = 20
            states = PBTController(pbt_cfg).run()
            self.assertEqual(len(states), 2)
            self.assertTrue((Path(pbt_cfg.run_dir) / "pbt_summary.json").exists())

    def test_live_monitor_state_reads_training_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = RunConfig(run_dir=str(Path(tmp) / "monitor"))
            cfg.train.steps = 20
            cfg.train.batch_size = 4
            cfg.train.log_every = 5
            result = Trainer(cfg).train()

            frame_dir = result.run_dir / "frames"
            frame_dir.mkdir(exist_ok=True)
            np.savez_compressed(frame_dir / "episode_9999.npz", frames=np.zeros((2, 4, 4), dtype=np.int8))

            state = collect_run_state(result.run_dir)

            self.assertTrue(state["exists"])
            self.assertEqual(state["summary"]["status"], "complete")
            self.assertEqual(state["summary"]["target_steps"], 20)
            self.assertGreaterEqual(state["metrics"]["row_count"], 1)
            self.assertIsNotNone(state["latest_frame"])
            self.assertEqual(state["latest_frame"]["grid"], [[0, 0, 0, 0]] * 4)


if __name__ == "__main__":
    unittest.main()
