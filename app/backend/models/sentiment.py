from transformers import pipeline

class SentimentClassifier:
    def __init__(self):
        self.pipe = pipeline(
            "text-classification",
            model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
            truncation=True,
        )

    def predict(self, aspect: str, sentence: str) -> str:
        # Feed aspect in context for targeted sentiment
        prompt = f"Regarding '{aspect}': {sentence}"
        result = self.pipe(prompt)[0]
        return result["label"].lower()  # positive / neutral / negative