from transformers import pipeline

CATEGORIES = [
    "food quality",
    "service",
    "price and value",
    "ambience and atmosphere",
    "location",
    "portion size",
    "menu variety",
    "cleanliness",
    "waiting time",
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
