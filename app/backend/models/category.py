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

    def predict_batch(self, aspects: list[str], sentence: str) -> list[str]:
        """
        Classifies every aspect of a review in a single batched forward pass
        instead of one pipeline call per aspect - the model dominates
        request latency, so batching is the main lever for a review with
        several aspects.
        """
        if not aspects:
            return []
        sequences = [f"{aspect} in context: {sentence}" for aspect in aspects]
        results = self.pipe(sequences, candidate_labels=CATEGORIES)
        if isinstance(results, dict):  # a single-sequence input isn't wrapped in a list
            results = [results]
        return [r["labels"][0].upper().replace(" ", "_") for r in results]
