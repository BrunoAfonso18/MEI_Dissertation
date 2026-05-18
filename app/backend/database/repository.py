from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from datetime import datetime, date
from database.models import (
    DimReview, 
    DimRestaurant, 
    DimCalendar, 
    FactSentiment
)


# ── Dimension Loading / ETL Functions ────────────────────────────

def load_or_create_dimensions(
    db: Session,
    restaurant_id: int,
    created_at: datetime
) -> dict:
    """
    Load or create dimension records for a review analysis.
    Returns dict with dim record IDs for use in fact table.
    """
    # Ensure calendar dimension exists for the date
    date_key = created_at.date()
    
    dim_calendar = db.query(DimCalendar).filter(
        DimCalendar.date_id == date_key
    ).first()
    
    if not dim_calendar:
        # Create new calendar entry
        dim_calendar = DimCalendar(
            date_id=date_key,
            date=date_key,
            week=date_key.replace(day=1),  # Simplified: use first day of week
            month=date_key.replace(day=1),
            year=date_key.year
        )
        db.add(dim_calendar)
        db.commit()
    
    # Note: restaurant_id should already exist in dim_restaurant
    # (populated during ETL from source systems)
    dim_restaurant = db.query(DimRestaurant).filter(
        DimRestaurant.id_restaurant == restaurant_id
    ).first()
    
    if not dim_restaurant:
        raise ValueError(f"Restaurant {restaurant_id} not found in dim_restaurant. Please populate dimensions first.")
    
    return {
        "date_id": dim_calendar.date_id,
        "restaurant_id": dim_restaurant.id_restaurant
    }


def create_dim_review(
    db: Session,
    review_id: int,
    text: str,
    source: str,
    language: str,
    created_at: datetime
) -> DimReview:
    """
    Create a review dimension record.
    Note: review_id should match source system ID.
    """
    dim_review = db.query(DimReview).filter(
        DimReview.id_review == review_id
    ).first()
    
    if dim_review:
        return dim_review
    
    dim_review = DimReview(
        id_review=review_id,
        text=text,
        source=source,
        language=language,
        created_at=created_at
    )
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
    inspection_grade: str = None
) -> DimRestaurant:
    """
    Create a restaurant dimension record.
    """
    dim_restaurant = db.query(DimRestaurant).filter(
        DimRestaurant.id_restaurant == restaurant_id
    ).first()
    
    if dim_restaurant:
        return dim_restaurant
    
    dim_restaurant = DimRestaurant(
        id_restaurant=restaurant_id,
        name=name,
        district=district,
        category=category,
        address=address,
        inspection_grade=inspection_grade
    )
    db.add(dim_restaurant)
    db.commit()
    db.refresh(dim_restaurant)
    return dim_restaurant


# ── Fact Table Functions ─────────────────────────────────────────

def save_fact_sentiment(
    db: Session,
    review_id: int,
    restaurant_id: int,
    created_at: datetime,
    aspect_data: dict
) -> FactSentiment:
    """
    Save sentiment analysis result to fact_sentiment table.
    
    aspect_data should contain:
    - aspect_term
    - opinion_term
    - aspect_category
    - pos_score (fuzzy positive)
    - neut_score (fuzzy neutral)
    - neg_score (fuzzy negative)
    - fuzzy_crisp_score (defuzzified score using centroid method)
    - sentiment_polarity
    - confidence
    """
    # Ensure review dimension exists
    dim_review = db.query(DimReview).filter(
        DimReview.id_review == review_id
    ).first()
    
    if not dim_review:
        raise ValueError(f"Review {review_id} not found in dim_review")
    
    # Ensure calendar dimension exists
    date_key = created_at.date()
    dim_calendar = db.query(DimCalendar).filter(
        DimCalendar.date_id == date_key
    ).first()
    
    if not dim_calendar:
        # Create it
        dim_calendar = DimCalendar(
            date_id=date_key,
            date=date_key,
            week=date_key.replace(day=1),
            month=date_key.replace(day=1),
            year=date_key.year
        )
        db.add(dim_calendar)
        db.commit()
    
    # Create fact record
    fact_sentiment = FactSentiment(
        id_review=review_id,
        id_restaurant=restaurant_id,
        date_id=date_key,
        aspect_term=aspect_data.get("aspect_term"),
        opinion_term=aspect_data.get("opinion_term"),
        aspect_category=aspect_data.get("aspect_category"),
        fuzzy_crisp_score=aspect_data.get("fuzzy_crisp_score"),  # Centroid method result
        sentiment_polarity=aspect_data.get("sentiment_polarity"),
        confidence_score=aspect_data.get("confidence"),
        created_at=created_at
    )
    db.add(fact_sentiment)
    db.commit()
    db.refresh(fact_sentiment)
    return fact_sentiment


# ── Analytical Queries ──────────────────────────────────────────

def sentiment_by_category(db: Session) -> list[dict]:
    """
    Count positive/neutral/negative per aspect category from fact table.
    """
    rows = (
        db.query(
            FactSentiment.aspect_category,
            FactSentiment.sentiment_polarity,
            func.count().label("count"),
            func.avg(FactSentiment.fuzzy_crisp_score).label("avg_crisp_score"),
        )
        .group_by(
            FactSentiment.aspect_category,
            FactSentiment.sentiment_polarity,
        )
        .order_by(desc("count"))
        .all()
    )
    return [
        {
            "category": r.aspect_category,
            "polarity": r.sentiment_polarity,
            "count": r.count,
            "avg_crisp_score": float(r.avg_crisp_score) if r.avg_crisp_score else None
        }
        for r in rows
    ]


def top_negative_aspects(db: Session, limit: int = 10) -> list[dict]:
    """
    Most complained about aspects from fact table.
    """
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
            "avg_crisp_score": float(r.avg_crisp_score) if r.avg_crisp_score else None
        }
        for r in rows
    ]


def sentiment_over_time(db: Session) -> list[dict]:
    """
    Sentiment trend grouped by day from fact table.
    """
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
            "avg_crisp_score": float(r.avg_crisp_score) if r.avg_crisp_score else None
        }
        for r in rows
    ]