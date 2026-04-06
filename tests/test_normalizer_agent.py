import unittest

from app.agents.normalizer_agent import NormalizationAgent


class NormalizerAgentTests(unittest.TestCase):
    def setUp(self):
        self.agent = NormalizationAgent()

    def test_merged_self_harm_phrase_is_repaired(self):
        result = self.agent.normalize("hey i am gonna killmyself")
        self.assertEqual(result["normalized_text"], "hey i am going to kill myself")
        self.assertTrue(result["changed"])
        self.assertGreaterEqual(result["fix_count"], 1)

    def test_harm_to_others_phrase_is_repaired(self):
        result = self.agent.normalize("imgonna killhim")
        self.assertIn("i am going to", result["normalized_text"])
        self.assertIn("kill him", result["normalized_text"])

    def test_clean_text_stays_unchanged(self):
        result = self.agent.normalize("I had coffee and went for a walk.")
        self.assertFalse(result["changed"])
        self.assertEqual(result["fix_count"], 0)


if __name__ == "__main__":
    unittest.main()
