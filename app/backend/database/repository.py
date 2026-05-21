from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from datetime import datetime, date
from database.models import (
    DimReview,
    DimRestaurant,
    DimCalendar,
    FactSentiment
)


# ── Calendar helpers ─────────────────────────────────────────────

def _ensure_calendar(db: Session, dt: datetime) -> DimCalendar:
    date_key = dt.date()
    dim_cal  = db.query(DimCalendar).filter(DimCalendar.date_id == date_key).first()
    if not dim_cal:
        dim_cal = DimCalendar(
            date_id=date_key,
            date=date_key,
            week=date_key.replace(day=1),
            month=date_key.replace(day=1),
            year=date_key.year,
        )
        db.add(dim_cal)
        db.commit()
    return dim_cal


# ── Dimension Loading / ETL Functions ────────────────────────────

def load_or_create_dimensions(
    db: Session,
    restaurant_id: int,
    created_at: datetime,
) -> dict:
    _ensure_calendar(db, created_at)
    dim_restaurant = db.query(DimRestaurant).filter(
        DimRestaurant.id_restaurant == restaurant_id
    ).first()
    if not dim_restaurant:
        raise ValueError(
            f"Restaurante {restaurant_id} não encontrado em dim_restaurant. "
            "Execute seed_restaurants.py primeiro."
        )
    return {
        "date_id": created_at.date(),
        "restaurant_id": dim_restaurant.id_restaurant,
    }


def create_dim_review(
    db: Session,
    text: str,
    source: str = "api",
    language: str = "pt",
    created_at: datetime = None,
    review_id: int = None,
) -> DimReview:
    """Create a review dimension record. ID is auto-generated when not provided."""
    if review_id is not None:
        existing = db.query(DimReview).filter(DimReview.id_review == review_id).first()
        if existing:
            return existing

    dim_review = DimReview(
        text=text,
        source=source,
        language=language,
        created_at=created_at or datetime.now(),
    )
    if review_id is not None:
        dim_review.id_review = review_id

    db.add(dim_review)
    db.commit()
    db.refresh(dim_review)
    return dim_review


def create_dim_restaurant(
    db: Session,
    restaurant_id: int,
    name: str,
    district: str = None,
    category: str = None,
    address: str = None,
    inspection_grade: str = None,
) -> DimRestaurant:
    existing = db.query(DimRestaurant).filter(
        DimRestaurant.id_restaurant == restaurant_id
    ).first()
    if existing:
        return existing

    dim_restaurant = DimRestaurant(
        id_restaurant=restaurant_id,
        name=name,
        district=district,
        category=category,
        address=address,
        inspection_grade=inspection_grade,
    )
    db.add(dim_restaurant)
    db.commit()
    db.refresh(dim_restaurant)
    return dim_restaurant


def get_all_restaurants(db: Session) -> list[DimRestaurant]:
    return db.query(DimRestaurant).order_by(DimRestaurant.id_restaurant).all()


# ── Fact Table Functions ─────────────────────────────────────────

def save_fact_sentiment(
    db: Session,
    review_id: int,
    restaurant_id: int,
    created_at: datetime,
    aspect_data: dict,
) -> FactSentiment:
    _ensure_calendar(db, created_at)

    fact = FactSentiment(
        id_review=review_id,
        id_restaurant=restaurant_id,
        date_id=created_at.date(),
        aspect_term=aspect_data.get("aspect_term"),
        opinion_term=aspect_data.get("opinion_term"),
        aspect_category=aspect_data.get("aspect_category"),
        fuzzy_crisp_score=aspect_data.get("fuzzy_crisp_score"),
        sentiment_polarity=aspect_data.get("sentiment_polarity"),
        confidence_score=aspect_data.get("confidence"),
        created_at=created_at,
    )
    db.add(fact)
    db.commit()
    db.refresh(fact)
    return fact


# ── Analytical Queries ───────────────────────────────────────────

def sentiment_by_category(db: Session) -> list[dict]:
    rows = (
        db.query(
            FactSentiment.aspect_category,
            FactSentiment.sentiment_polarity,
            func.count().label("count"),
            func.avg(FactSentiment.fuzzy_crisp_score).label("avg_crisp_score"),
        )
        .group_by(FactSentiment.aspect_category, FactSentiment.sentiment_polarity)
        .order_by(desc("count"))
        .all()
    )
    return [
        {
            "category": r.aspect_category,
            "polarity": r.sentiment_polarity,
            "count": r.count,
            "avg_crisp_score": float(r.avg_crisp_score) if r.avg_crisp_score else None,
        }
        for r in rows
    ]


def top_negative_aspects(db: Session, limit: int = 10) -> list[dict]:
    rows = (
        db.query(
            FactSentiment.aspect_term,
            func.count().label("count"),
            func.avg(FactSentiment.fuzzy_crisp_score).label("avg_crisp_score"),
        )
        .filter(FactSentiment.sentiment_polarity == "negative")
        .group_by(FactSentiment.aspect_term)
        .order_by(desc("count"))
        .limit(limit)
        .all()
    )
    return [
        {
            "aspect": r.aspect_term,
            "count": r.count,
            "avg_crisp_score": float(r.avg_crisp_score) if r.avg_crisp_score else None,
        }
        for r in rows
    ]


def sentiment_over_time(db: Session) -> list[dict]:
    rows = (
        db.query(
            FactSentiment.date_id,
            FactSentiment.sentiment_polarity,
            func.count().label("count"),
            func.avg(FactSentiment.fuzzy_crisp_score).label("avg_crisp_score"),
        )
        .group_by(FactSentiment.date_id, FactSentiment.sentiment_polarity)
        .order_by(FactSentiment.date_id)
        .all()
    )
    return [
        {
            "date": str(r.date_id),
            "polarity": r.sentiment_polarity,
            "count": r.count,
            "avg_crisp_score": float(r.avg_crisp_score) if r.avg_crisp_score else None,
        }
        for r in rows
    ]
