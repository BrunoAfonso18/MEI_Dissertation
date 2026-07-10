"""
Bridges the desktop frontend to this project's own ABSA backend.

Reuses the exact same aspect extraction model and fuzzy-logic sentiment
scoring that the FastAPI backend (app/backend/main.py) uses, so the smiley
reacts to the same pipeline that powers the rest of the dissertation project.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from models.extractor import AspectExtractor  # noqa: E402
from models.fuzzy_sentiment import fuzzy_analyzer, polarity_to_scores  # noqa: E402

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "absa_module" / "absa_model_final_polarity"


@dataclass
class AspectSentiment:
    term: str
    polarity: str
    confidence: float


@dataclass
class SentimentResult:
    positivity: float  # in [-1, 1], for driving the smiley
    label: str
    aspects: list[AspectSentiment] = field(default_factory=list)


class AbsaSentimentPipeline:
    """
    Runs the project's ABSA extractor + fuzzy analyzer end-to-end and
    collapses the result into a single positivity score for the smiley.
    """

    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH) -> None:
        self.extractor = AspectExtractor(str(model_path))

    def run(self, text: str) -> SentimentResult:
        if not text.strip():
            return SentimentResult(positivity=0.0, label="neutral")

        extracted = self.extractor.predict(text)
        aspects = extracted["aspects"]
        polarities = extracted["polarities"]
        confidences = extracted["confidences"]

        if not aspects:
            return SentimentResult(positivity=0.0, label="neutral")

        fuzzy_results = []
        aspect_sentiments = []
        for aspect, polarity, confidence in zip(aspects, polarities, confidences):
            raw_scores = polarity_to_scores(polarity or "neutral", confidence)
            fuzzy_result = fuzzy_analyzer.analyze(raw_scores)
            fuzzy_results.append(fuzzy_result)
            aspect_sentiments.append(
                AspectSentiment(
                    term=aspect,
                    polarity=fuzzy_result["sentiment_label"],
                    confidence=confidence,
                )
            )

        aggregated = fuzzy_analyzer.aggregate_aspects(fuzzy_results)
        positivity = (aggregated["defuzzified_score"] - 0.5) * 2

        return SentimentResult(
            positivity=positivity,
            label=aggregated["sentiment_label"],
            aspects=aspect_sentiments,
        )
