import json
import tempfile
import unittest
from pathlib import Path

from arcagent.eval import run_benchmark


class BenchmarkTests(unittest.TestCase):
    def test_smoke_benchmark_writes_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_benchmark(
                tmp,
                seed=3,
                episodes=1,
                train_steps=8,
                pbt_generations=1,
                pbt_population=2,
                pbt_steps_per_generation=8,
                log_every=4,
            )
            path = Path(tmp) / "benchmark.json"
            self.assertTrue(path.exists())
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["seed"], 3)
            self.assertEqual([row["name"] for row in result["benchmark"]], ["untrained", "single_train", "tiny_pbt"])
            for row in result["benchmark"]:
                self.assertIn("eval_return_mean", row)
                self.assertIn("eval_success_rate", row)


if __name__ == "__main__":
    unittest.main()
