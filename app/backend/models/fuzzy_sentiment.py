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
        if not fuzzy_scores:
            return 0.5

        neg = fuzzy_scores.get("negative", 0.0)
        neu = fuzzy_scores.get("neutral", 0.0)
        pos = fuzzy_scores.get("positive", 0.0)

        total = neg + neu + pos
        if total == 0.0:
            return 0.5

        return (neg * 0.0 + neu * 0.5 + pos * 1.0) / total

    def defuzzify_label(self, fuzzy_scores: dict) -> str:
        scores = {
            "negative": fuzzy_scores["negative"],
            "neutral": fuzzy_scores["neutral"],
            "positive": fuzzy_scores["positive"],
        }
        return max(scores, key=scores.get)

    def analyze(self, raw_scores: dict) -> dict:
        pos = raw_scores.get("positive_score", 0.33)
        neu = raw_scores.get("neutral_score", 0.33)
        neg = raw_scores.get("negative_score", 0.34)

        defuzz = self.defuzzify_centroid({"negative": neg, "neutral": neu, "positive": pos})
        label = self.defuzzify_label({"negative": neg, "neutral": neu, "positive": pos})

        return {
            "positive_score": float(pos),
            "neutral_score": float(neu),
            "negative_score": float(neg),
            "defuzzified_score": float(defuzz),
            "sentiment_label": label,
        }

    def aggregate_aspects(self, aspect_results: list[dict]) -> dict:
        if not aspect_results:
            return {"positive": 0.0, "neutral": 0.0, "negative": 0.0}

        avg_positive = np.mean([r["positive_score"] for r in aspect_results])
        avg_neutral = np.mean([r["neutral_score"] for r in aspect_results])
        avg_negative = np.mean([r["negative_score"] for r in aspect_results])

        if avg_positive == 0.0 and avg_neutral == 0.0 and avg_negative == 0.0:
            label = "neutral"
            defuzz_score = 0.5
        else:
            fuzzy_scores = {
                "negative": avg_negative,
                "neutral": avg_neutral,
                "positive": avg_positive,
            }
            label = self.defuzzify_label(fuzzy_scores)
            defuzz_score = self.defuzzify_centroid(fuzzy_scores)

        return {
            "positive": float(avg_positive),
            "neutral": float(avg_neutral),
            "negative": float(avg_negative),
            "defuzzified_score": float(defuzz_score),
            "sentiment_label": label,
        }


fuzzy_analyzer = FuzzySentimentAnalyzer()