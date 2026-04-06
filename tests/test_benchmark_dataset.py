import unittest
from pathlib import Path

import pandas as pd


class BenchmarkDatasetTests(unittest.TestCase):
    def test_benchmark_dataset_has_valid_schema_and_labels(self):
        path = Path("data/benchmarks/system_benchmark.csv")
        self.assertTrue(path.exists())

        df = pd.read_csv(path)
        self.assertTrue({"text", "expected_label", "category", "notes"}.issubset(df.columns))
        self.assertGreaterEqual(len(df), 200)

        valid_labels = {
            "LOW_RISK",
            "MODERATE_RISK",
            "HIGH_RISK_SELF_HARM",
            "HIGH_RISK_HARM_TO_OTHERS",
        }
        self.assertEqual(set(df["expected_label"]) - valid_labels, set())
        self.assertGreaterEqual(df["category"].nunique(), 10)


if __name__ == "__main__":
    unittest.main()
