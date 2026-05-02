import numpy as np
import skfuzzy as fuzz
from skfuzzy import membership as mem


class FuzzySentimentAnalyzer:
    def __init__(self):
        self.x_confidence = np.linspace(0, 1, 100)
        self.negative = self._create_triangular(0.0, 0.0, 0.5)
        self.neutral = self._create_triangular(0.2, 0.5, 0.8)
        self.positive = self._create_triangular(0.5, 1.0, 1.0)

    def _create_triangular(self, a: float, b: float, c: float) -> np.ndarray:
        return fuzz.trimf(self.x_confidence, [a, b, c])

    def fuzzify(self, confidence_scores: dict) -> dict:
        conf = confidence_scores.get("confidence", 0.5)
        return {
            "negative": fuzz.interp_membership(self.x_confidence, self.negative, conf),
            "neutral": fuzz.interp_membership(self.x_confidence, self.neutral, conf),
            "positive": fuzz.interp_membership(self.x_confidence, self.positive, conf),
        }

    def defuzzify_centroid(self, fuzzy_scores: dict) -> float:
        aggregated = np.fmax(
            fuzzy_scores["negative"],
            np.fmax(fuzzy_scores["neutral"], fuzzy_scores["positive"])
        )
        return fuzz.defuzz(self.x_confidence, aggregated, "centroid")

    def defuzzify_label(self, fuzzy_scores: dict) -> str:
        scores = {
            "negative": fuzzy_scores["negative"],
            "neutral": fuzzy_scores["neutral"],
            "positive": fuzzy_scores["positive"],
        }
        return max(scores, key=scores.get)

    def analyze(self, raw_scores: dict) -> dict:
        raw_conf = raw_scores.get("confidence", 0.5)
        
        fuzzy_scores = self.fuzzify({"confidence": raw_conf})
        
        return {
            "positive_score": float(fuzzy_scores["positive"]),
            "neutral_score": float(fuzzy_scores["neutral"]),
            "negative_score": float(fuzzy_scores["negative"]),
            "defuzzified_score": float(self.defuzzify_centroid(fuzzy_scores)),
            "sentiment_label": self.defuzzify_label(fuzzy_scores),
        }

    def aggregate_aspects(self, aspect_results: list[dict]) -> dict:
        if not aspect_results:
            return {"positive": 0.0, "neutral": 0.0, "negative": 0.0}

        avg_positive = np.mean([r["positive_score"] for r in aspect_results])
        avg_neutral = np.mean([r["neutral_score"] for r in aspect_results])
        avg_negative = np.mean([r["negative_score"] for r in aspect_results])

        aggregated = {
            "positive": float(avg_positive),
            "neutral": float(avg_neutral),
            "negative": float(avg_negative),
        }

        return {
            **aggregated,
            "defuzzified_score": float(self.defuzzify_centroid(aggregated)),
            "sentiment_label": self.defuzzify_label(aggregated),
        }


fuzzy_analyzer = FuzzySentimentAnalyzer()