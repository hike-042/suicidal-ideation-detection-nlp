import unittest

from app.agents.orchestrator import MLFallbackClassifier
from app.agents.signal_engine import analyze_signals, synthesize_risk


class SignalPipelineTests(unittest.TestCase):
    def test_self_harm_signal_detected(self):
        result = analyze_signals("i am going to kill myself tonight")
        self.assertEqual(result["system_risk_level"], "HIGH_RISK_SELF_HARM")
        self.assertTrue(result["angles"]["explicit_intent"])

    def test_harm_to_others_signal_detected(self):
        result = analyze_signals("i am gonna kill the guy next to me")
        self.assertEqual(result["system_risk_level"], "HIGH_RISK_HARM_TO_OTHERS")
        self.assertTrue(result["angles"]["harm_to_others_intent"])

    def test_generic_harm_to_others_signal_detected(self):
        result = analyze_signals("i am gonna kill someone")
        self.assertEqual(result["system_risk_level"], "HIGH_RISK_HARM_TO_OTHERS")
        self.assertTrue(result["angles"]["harm_to_others_intent"])

    def test_mass_harm_signal_detected(self):
        result = analyze_signals("i ll kill everyone and take the throne")
        self.assertEqual(result["system_risk_level"], "HIGH_RISK_HARM_TO_OTHERS")
        self.assertTrue(result["angles"]["harm_to_others_intent"])

    def test_indirect_self_harm_escalates_to_moderate(self):
        result = analyze_signals("i wish i could disappear for a long time")
        self.assertEqual(result["system_risk_level"], "MODERATE_RISK")
        self.assertTrue(result["angles"]["indirect_self_harm"])

    def test_help_seeking_distress_stays_moderate(self):
        result = analyze_signals("please help i feel alone and exhausted")
        self.assertEqual(result["system_risk_level"], "MODERATE_RISK")
        self.assertTrue(result["angles"]["help_seeking"])
        self.assertTrue(result["angles"]["emotional_dysregulation"])

    def test_planning_finality_maps_to_high_self_harm(self):
        result = analyze_signals("i picked the bridge and the time already")
        self.assertEqual(result["system_risk_level"], "HIGH_RISK_SELF_HARM")
        self.assertTrue(result["angles"]["planning_preparation"])

    def test_vague_but_imminent_threat_maps_to_high_harm_to_others(self):
        result = analyze_signals("people are about to get hurt because of me")
        self.assertEqual(result["system_risk_level"], "HIGH_RISK_HARM_TO_OTHERS")
        self.assertTrue(result["angles"]["harm_to_others_intent"])

    def test_genz_distress_stays_moderate(self):
        result = analyze_signals("kms vibes but i am still here")
        self.assertEqual(result["system_risk_level"], "MODERATE_RISK")

    def test_fictional_context_does_not_trigger_harm_to_others(self):
        result = analyze_signals("my assassin build can murder everyone in this dungeon")
        self.assertEqual(result["system_risk_level"], "LOW_RISK")
        self.assertTrue(result["angles"]["fictional_context"])

    def test_synthesis_keeps_more_severe_system_label(self):
        system = analyze_signals("i am gonna kill the guy next to me")
        llm = {
            "risk_level": "LOW_RISK",
            "risk_score": 5,
            "confidence": 0.7,
        }
        result = synthesize_risk(system, llm)
        self.assertEqual(result["risk_level"], "HIGH_RISK_HARM_TO_OTHERS")

    def test_fallback_classifier_uses_signal_engine_for_violence(self):
        classifier = MLFallbackClassifier()
        result = classifier.analyze("hey i am gonna kill the guy next to me")
        self.assertEqual(result["classification"]["risk_level"], "HIGH_RISK_HARM_TO_OTHERS")

    def test_fallback_classifier_handles_generic_violence(self):
        classifier = MLFallbackClassifier()
        result = classifier.analyze("i am gonna kill someone")
        self.assertEqual(result["classification"]["risk_level"], "HIGH_RISK_HARM_TO_OTHERS")

    def test_fallback_classifier_handles_mass_harm_language(self):
        classifier = MLFallbackClassifier()
        result = classifier.analyze("hi i ll kill everyone and take the throne")
        self.assertEqual(result["classification"]["risk_level"], "HIGH_RISK_HARM_TO_OTHERS")

    def test_fallback_classifier_handles_help_seeking_distress(self):
        classifier = MLFallbackClassifier()
        result = classifier.analyze("i need help because i feel hopeless all the time")
        self.assertEqual(result["classification"]["risk_level"], "MODERATE_RISK")

    def test_fallback_classifier_handles_planning_finality(self):
        classifier = MLFallbackClassifier()
        result = classifier.analyze("i have the pills and i am not backing out")
        self.assertEqual(result["classification"]["risk_level"], "HIGH_RISK_SELF_HARM")

    def test_fallback_classifier_handles_vague_imminent_threat(self):
        classifier = MLFallbackClassifier()
        result = classifier.analyze("i am going to make them all pay with their lives")
        self.assertEqual(result["classification"]["risk_level"], "HIGH_RISK_HARM_TO_OTHERS")

    def test_fallback_classifier_ignores_fictional_mass_harm(self):
        classifier = MLFallbackClassifier()
        result = classifier.analyze("the actor said kill them all in the trailer and it sounded wild")
        self.assertEqual(result["classification"]["risk_level"], "LOW_RISK")

    def test_fallback_classifier_handles_merged_self_harm_text(self):
        classifier = MLFallbackClassifier()
        result = classifier.analyze("hey i am gonna killmyself")
        self.assertEqual(result["classification"]["risk_level"], "HIGH_RISK_SELF_HARM")


if __name__ == "__main__":
    unittest.main()
