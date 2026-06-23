import unittest

from arcagent.envs import MockArcEnv


class EnvTests(unittest.TestCase):
    def test_mock_env_shapes_and_rewards(self):
        env = MockArcEnv(grid_size=16, max_steps=20, seed=0)
        obs = env.reset()
        self.assertEqual(obs["grid"].shape, (16, 16))
        next_obs, reward, done, info = env.step(4)
        self.assertEqual(next_obs["grid"].shape, (16, 16))
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIn("available_actions", next_obs)


if __name__ == "__main__":
    unittest.main()
