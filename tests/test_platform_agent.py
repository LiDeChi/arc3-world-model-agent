import tempfile
import unittest
from pathlib import Path

from arcagent.config import RunConfig
from arcagent.platform.agent import PlatformAgent
from arcagent.platform.llm_client import LLMResult, extract_json_object


class FakeClient:
    def chat(self, messages, models, temperature=0.2, max_tokens=700):
        return LLMResult(
            ok=True,
            model="fake-deepseek",
            content='{"rationale":"test proposal","changes":{"train.epsilon":0.21,"model.planning_rollouts":10},"expected_effect":"better exploration","risk":"short run noise"}',
        )


class PlatformAgentTests(unittest.TestCase):
    def test_extract_json_from_reasoning_text(self):
        text = 'Thinking... final JSON is {"rationale":"x","changes":{"train.epsilon":0.2},"expected_effect":"y","risk":"z"}'
        obj = extract_json_object(text)
        self.assertEqual(obj["changes"]["train.epsilon"], 0.2)

    def test_platform_agent_runs_one_fake_llm_cycle(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = RunConfig(run_dir=str(Path(tmp) / "platform"))
            cfg.train.batch_size = 4
            agent = PlatformAgent(cfg, repo_root=Path(__file__).resolve().parents[1], llm_client=FakeClient(), models=["fake-deepseek"])
            outcomes = agent.run(cycles=1, steps_per_cycle=25)
            self.assertEqual(len(outcomes), 1)
            self.assertTrue(outcomes[0].llm_ok)
            self.assertEqual(outcomes[0].proposal.changes["train.epsilon"], 0.21)
            self.assertTrue(Path(outcomes[0].report_path).exists())
            self.assertTrue((Path(cfg.run_dir) / "platform_agent_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
