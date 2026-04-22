from transformers import pipeline

CATEGORIES = [
    "food quality", "food price", "service", "ambience",
    "battery life", "camera quality", "screen quality",
    "performance", "design", "value for money", "software",
]

class CategoryClassifier:
    def __init__(self):
        self.pipe = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
        )

    def predict(self, aspect: str, sentence: str) -> str:
        result = self.pipe(
            f"{aspect} in context: {sentence}",
            candidate_labels=CATEGORIES,
        )
        return result["labels"][0].upper().replace(" ", "_")