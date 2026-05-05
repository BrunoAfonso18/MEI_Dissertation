from transformers import pipeline

class SentimentClassifier:
    def __init__(self):
        self.pipe = pipeline(
            "text-classification",
            model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
            truncation=True,
            return_all_scores=True,
        )

    def predict(self, aspect: str, sentence: str) -> dict:
        prompt = f"Regarding '{aspect}': {sentence}"
        results = self.pipe(prompt, top_k=None)

        if isinstance(results, list) and len(results) > 0:
            first_result = results[0]
            if isinstance(first_result, dict) and "label" in first_result:
                scores = {r["label"].lower(): r["score"] for r in results}
                label = max(scores, key=scores.get)
                return {
                    "label": label,
                    "confidence": scores.get(label, 0.0),
                    "positive_score": scores.get("positive", 0.0),
                    "neutral_score": scores.get("neutral", 0.0),
                    "negative_score": scores.get("negative", 0.0),
                }

        if isinstance(results, list) and len(results) > 0 and isinstance(results[0], dict) and "score" in results[0]:
            label = results[0].get("label", "neutral").lower()
            score = results[0].get("score", 0.0)
            all_scores = {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
            all_scores[label] = score
            return {
                "label": label,
                "confidence": score,
                "positive_score": all_scores["positive"],
                "neutral_score": all_scores["neutral"],
                "negative_score": all_scores["negative"],
            }

        return {
            "label": "neutral",
            "confidence": 0.0,
            "positive_score": 0.0,
            "neutral_score": 1.0,
            "negative_score": 0.0,
        }