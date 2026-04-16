from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from database.models import Review, AspectPrediction


def save_review(db: Session, text: str, source: str = "api", language: str = "en") -> Review:
    review = Review(text=text, source=source, language=language)
    db.add(review)
    db.commit()
    db.refresh(review)
    return review


def save_aspects(db: Session, review_id: int, aspects: list[dict]) -> list[AspectPrediction]:
    predictions = [
        AspectPrediction(
            review_id          = review_id,
            aspect_term        = a["aspect_term"],
            opinion_term       = a.get("opinion_term"),
            sentiment_polarity = a["sentiment_polarity"],
            aspect_category    = a["aspect_category"],
        )
        for a in aspects
    ]
    db.add_all(predictions)
    db.commit()
    return predictions


# ── Analytical Queries ──────────────────────────────────────────

def sentiment_by_category(db: Session) -> list[dict]:
    """Count positive/neutral/negative per aspect category"""
    rows = (
        db.query(
            AspectPrediction.aspect_category,
            AspectPrediction.sentiment_polarity,
            func.count().label("count"),
        )
        .group_by(
            AspectPrediction.aspect_category,
            AspectPrediction.sentiment_polarity,
        )
        .order_by(desc("count"))
        .all()
    )
    return [
        {"category": r.aspect_category, "polarity": r.sentiment_polarity, "count": r.count}
        for r in rows
    ]


def top_negative_aspects(db: Session, limit: int = 10) -> list[dict]:
    """Most complained about aspects"""
    rows = (
        db.query(
            AspectPrediction.aspect_term,
            func.count().label("count"),
        )
        .filter(AspectPrediction.sentiment_polarity == "negative")
        .group_by(AspectPrediction.aspect_term)
        .order_by(desc("count"))
        .limit(limit)
        .all()
    )
    return [{"aspect": r.aspect_term, "count": r.count} for r in rows]


def sentiment_over_time(db: Session) -> list[dict]:
    """Sentiment trend grouped by day"""
    rows = (
        db.query(
            func.date_trunc("day", AspectPrediction.created_at).label("day"),
            AspectPrediction.sentiment_polarity,
            func.count().label("count"),
        )
        .group_by("day", AspectPrediction.sentiment_polarity)
        .order_by("day")
        .all()
    )
    return [
        {"day": str(r.day), "polarity": r.sentiment_polarity, "count": r.count}
        for r in rows
    ]