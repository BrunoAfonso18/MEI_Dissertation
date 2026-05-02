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
        results = self.pipe(prompt)[0]

        scores = {r["label"].lower(): r["score"] for r in results}
        
        label = max(scores, key=scores.get)

        return {
            "label": label,
            "confidence": scores.get(label, 0.0),
            "all_scores": scores,
        }