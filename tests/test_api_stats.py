import unittest

from app.api import routes


class ApiStatsTests(unittest.TestCase):
    def test_record_analysis_counts_high_risk_variants(self):
        original_counts = routes._stats["risk_level_counts"].copy()
        try:
            for key in routes._stats["risk_level_counts"]:
                routes._stats["risk_level_counts"][key] = 0

            routes._record_analysis(
                {
                    "classification": {"risk_level": "HIGH_RISK_SELF_HARM"},
                    "processing_time_ms": 12,
                    "tier_used": "haiku",
                }
            )
            routes._record_analysis(
                {
                    "classification": {"risk_level": "HIGH_RISK_HARM_TO_OTHERS"},
                    "processing_time_ms": 12,
                    "tier_used": "haiku",
                }
            )

            self.assertEqual(routes._stats["risk_level_counts"]["HIGH_RISK"], 2)
            self.assertEqual(routes._stats["risk_level_counts"]["HIGH_RISK_SELF_HARM"], 1)
            self.assertEqual(routes._stats["risk_level_counts"]["HIGH_RISK_HARM_TO_OTHERS"], 1)
        finally:
            routes._stats["risk_level_counts"].update(original_counts)


if __name__ == "__main__":
    unittest.main()
