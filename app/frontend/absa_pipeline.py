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
from models.fuzzy_inference import IMPLICIT_ASPECT, label_for, score_aspect, score_implicit  # noqa: E402

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
        opinions = extracted["opinions"]
        polarities = extracted["polarities"]
        confidences = extracted["confidences"]

        if not aspects:
            # Mesma opinião global que o backend guarda no aspeto implícito "geral".
            implicit = score_implicit(text)
            if implicit is None:
                return SentimentResult(positivity=0.0, label="neutral")
            return SentimentResult(
                positivity=(implicit["crisp"] - 0.5) * 2,
                label=implicit["label"],
                aspects=[AspectSentiment(term=IMPLICIT_ASPECT, polarity=implicit["label"],
                                         confidence=implicit["certainty"])],
            )

        # Same per-aspect scoring as the backend (main.py), so the smiley shows
        # exactly the value that is stored in the data warehouse.
        scores = []
        aspect_sentiments = []
        for i, (aspect, polarity, confidence) in enumerate(zip(aspects, polarities, confidences)):
            opinion = opinions[i] if i < len(opinions) else None
            scored = score_aspect(text, aspect, opinion, polarity, confidence)
            scores.append(scored["crisp"])
            aspect_sentiments.append(
                AspectSentiment(
                    term=aspect,
                    polarity=scored["label"],
                    confidence=scored["certainty"],
                )
            )

        # Review sentiment = simple mean of the aspects' intensities.
        mean_score = sum(scores) / len(scores)
        positivity = (mean_score - 0.5) * 2

        return SentimentResult(
            positivity=positivity,
            label=label_for(mean_score),
            aspects=aspect_sentiments,
        )
