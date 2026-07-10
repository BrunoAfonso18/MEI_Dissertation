import numpy as np
import skfuzzy as fuzz
from skfuzzy import membership as mem


def polarity_to_scores(polarity: str, confidence: float) -> dict:
    """Convert a BIO polarity label + softmax confidence into fuzzy input scores."""
    conf = max(0.0, min(1.0, confidence))
    rest = (1.0 - conf) / 2.0
    if polarity == "positive":
        return {"positive_score": conf, "neutral_score": rest, "negative_score": rest}
    if polarity == "negative":
        return {"positive_score": rest, "neutral_score": rest, "negative_score": conf}
    if polarity == "neutral":
        return {"positive_score": rest, "neutral_score": conf, "negative_score": rest}
    # conflict
    return {"positive_score": 0.4, "neutral_score": 0.2, "negative_score": 0.4}


class FuzzySentimentAnalyzer:
    """
    Fuzzy Logic Sentiment Analyzer for ABSA.
    
    Converts raw sentiment scores (positive, neutral, negative) to fuzzy membership
    values, then defuzzifies to get a crisp score using the centroid method.
    
    The crisp score is a single value in [0, 1] where:
    - 0.0 = negative sentiment
    - 0.5 = neutral sentiment
    - 1.0 = positive sentiment
    """
    
    def __init__(self):
        # Universe of discourse for fuzzy membership
        self.x_confidence = np.linspace(0, 1, 100)
        
        # Define triangular membership functions for sentiment classes
        self.negative = self._create_triangular(0.0, 0.0, 0.5)  # Peak at 0 (negative)
        self.neutral = self._create_triangular(0.2, 0.5, 0.8)    # Peak at 0.5 (neutral)
        self.positive = self._create_triangular(0.5, 1.0, 1.0)  # Peak at 1 (positive)

    def _create_triangular(self, a: float, b: float, c: float) -> np.ndarray:
        """
        Create a triangular membership function.
        Parameters:
            a: left slope start
            b: peak (mode)
            c: right slope end
        """
        return fuzz.trimf(self.x_confidence, [a, b, c])

    def fuzzify(self, confidence_scores: dict) -> dict:
        """
        Fuzzify input scores: convert crisp values to fuzzy membership values.
        
        Input confidence_scores should have 'confidence' key with value in [0, 1].
        Returns fuzzy membership values for each sentiment class.
        """
        conf = confidence_scores.get("confidence", 0.5)
        return {
            "negative": fuzz.interp_membership(self.x_confidence, self.negative, conf),
            "neutral": fuzz.interp_membership(self.x_confidence, self.neutral, conf),
            "positive": fuzz.interp_membership(self.x_confidence, self.positive, conf),
        }

    def defuzzify_centroid(self, fuzzy_scores: dict) -> float:
        """
        Defuzzify using the centroid method.
        
        Formula: crisp_value = (0.0 * neg_score + 0.5 * neu_score + 1.0 * pos_score) / (neg_score + neu_score + pos_score)
        
        This produces a single crisp value in [0, 1] representing the overall sentiment,
        weighted by the fuzzy membership degrees.
        
        Returns:
            float: Crisp sentiment score in [0, 1]
        """
        if not fuzzy_scores:
            return 0.5

        neg = fuzzy_scores.get("negative", 0.0)
        neu = fuzzy_scores.get("neutral", 0.0)
        pos = fuzzy_scores.get("positive", 0.0)

        total = neg + neu + pos
        if total == 0.0:
            return 0.5

        # Centroid formula: weighted average of sentiment class centers
        crisp_value = (neg * 0.0 + neu * 0.5 + pos * 1.0) / total
        return float(crisp_value)

    def defuzzify_label(self, fuzzy_scores: dict) -> str:
        """
        Defuzzify to a linguistic label by selecting the class with highest membership.
        
        Returns: "negative", "neutral", or "positive"
        """
        scores = {
            "negative": fuzzy_scores["negative"],
            "neutral": fuzzy_scores["neutral"],
            "positive": fuzzy_scores["positive"],
        }
        return max(scores, key=scores.get)

    def analyze(self, raw_scores: dict) -> dict:
        """
        Complete sentiment analysis pipeline.
        
        1. Extract raw sentiment scores (pos, neu, neg) - these are the fuzzy membership values
        2. Calculate crisp score using centroid method
        3. Determine sentiment label (max membership)
        
        Input raw_scores should contain:
        - positive_score: fuzzy membership in positive class
        - neutral_score: fuzzy membership in neutral class
        - negative_score: fuzzy membership in negative class
        
        Returns dict with:
        - positive_score: fuzzy positive membership
        - neutral_score: fuzzy neutral membership
        - negative_score: fuzzy negative membership
        - defuzzified_score: crisp value [0, 1] using centroid method
        - sentiment_label: linguistic label (positive, neutral, negative)
        """
        pos = raw_scores.get("positive_score", 0.33)
        neu = raw_scores.get("neutral_score", 0.33)
        neg = raw_scores.get("negative_score", 0.34)

        # Calculate crisp score using centroid method
        crisp_score = self.defuzzify_centroid({"negative": neg, "neutral": neu, "positive": pos})
        
        # Get linguistic label
        label = self.defuzzify_label({"negative": neg, "neutral": neu, "positive": pos})

        return {
            "positive_score": float(pos),      # Fuzzy value
            "neutral_score": float(neu),       # Fuzzy value
            "negative_score": float(neg),      # Fuzzy value
            "defuzzified_score": crisp_score,  # Crisp value from centroid method
            "sentiment_label": label,          # Linguistic label
        }

    def aggregate_aspects(self, aspect_results: list[dict]) -> dict:
        """
        Aggregate fuzzy scores across multiple aspects in a single review.
        
        Computes averages of fuzzy memberships, then applies centroid defuzzification.
        
        Returns aggregated sentiment metrics for the entire review.
        """
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