import unittest

from arcagent.agent import WorldModelAgent
from arcagent.data import Transition, ReplayBuffer
from arcagent.envs import MockArcEnv


class ModelAgentTests(unittest.TestCase):
    def test_agent_acts_and_trains_world_model(self):
        env = MockArcEnv(grid_size=16, max_steps=10, seed=1)
        agent = WorldModelAgent.create(grid_size=16, seed=1)
        obs = env.reset()
        action = agent.act(obs)
        self.assertIn(action, [1, 2, 3, 4, 5])
        next_obs, reward, done, _info = env.step(action)
        buf = ReplayBuffer(seed=1)
        buf.add(Transition(obs["grid"], action, reward, next_obs["grid"], done))
        metrics = agent.train_world_model(buf.sample(1), learning_rate=1e-3)
        self.assertIn("world_model_loss", metrics)
        imagined = agent.imagined_rollout(obs["grid"], steps=3)
        self.assertEqual(len(imagined), 3)
        self.assertEqual(imagined[0].shape, obs["grid"].shape)


if __name__ == "__main__":
    unittest.main()
