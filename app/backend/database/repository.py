from dataclasses import dataclass, field
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from datetime import datetime, date
from database.models import (
    DimReview,
    DimRestaurant,
    DimCalendar,
    FactSentiment
)


@dataclass
class AnalyticsFilters:
    """
    The dashboard filter sidebar's full filter set, built from query params
    and threaded through every /analytics/* query. Every list field is an
    OR-of-selected-values (SQL IN) filter; an empty list means "no filter on
    that field".
    """
    start_date: date = None
    end_date: date = None
    restaurant_ids: list[int] = field(default_factory=list)
    districts: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)         # DimRestaurant.category (cuisine)
    grades: list[str] = field(default_factory=list)             # DimRestaurant.inspection_grade
    polarities: list[str] = field(default_factory=list)         # FactSentiment.sentiment_polarity
    aspect_categories: list[str] = field(default_factory=list)  # FactSentiment.aspect_category
    aspect_terms: list[str] = field(default_factory=list)       # FactSentiment.aspect_term (drill-down)


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

def add_fact_sentiment(
    db: Session,
    review_id: int,
    restaurant_id: int,
    created_at: datetime,
    aspect_data: dict,
) -> FactSentiment:
    """
    Stages a fact row without committing. Callers that insert several facts
    for the same review (e.g. one per aspect) should call this in a loop and
    commit once at the end instead of round-tripping per row.
    """
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
    return fact


def save_fact_sentiment(
    db: Session,
    review_id: int,
    restaurant_id: int,
    created_at: datetime,
    aspect_data: dict,
) -> FactSentiment:
    _ensure_calendar(db, created_at)
    fact = add_fact_sentiment(db, review_id, restaurant_id, created_at, aspect_data)
    db.commit()
    db.refresh(fact)
    return fact


# ── Analytical Queries ───────────────────────────────────────────

def _apply_fact_filters(
    query,
    f: AnalyticsFilters,
    joined_restaurant: bool = False,
):
    """
    Applies the dashboard filter sidebar's full filter set to a query
    already selecting from FactSentiment. Joins DimRestaurant only when a
    restaurant-level filter (district/category/grade) needs it and the
    caller hasn't already joined it.
    """
    if (f.districts or f.categories or f.grades) and not joined_restaurant:
        query = query.join(DimRestaurant, DimRestaurant.id_restaurant == FactSentiment.id_restaurant)
        joined_restaurant = True

    if f.start_date:
        query = query.filter(FactSentiment.date_id >= f.start_date)
    if f.end_date:
        query = query.filter(FactSentiment.date_id <= f.end_date)
    if f.restaurant_ids:
        query = query.filter(FactSentiment.id_restaurant.in_(f.restaurant_ids))
    if f.polarities:
        query = query.filter(FactSentiment.sentiment_polarity.in_(f.polarities))
    if f.aspect_categories:
        query = query.filter(FactSentiment.aspect_category.in_(f.aspect_categories))
    if f.aspect_terms:
        query = query.filter(FactSentiment.aspect_term.in_(f.aspect_terms))
    if f.districts:
        query = query.filter(DimRestaurant.district.in_(f.districts))
    if f.categories:
        query = query.filter(DimRestaurant.category.in_(f.categories))
    if f.grades:
        query = query.filter(DimRestaurant.inspection_grade.in_(f.grades))
    return query


def sentiment_by_category(db: Session, filters: AnalyticsFilters = None) -> list[dict]:
    f = filters or AnalyticsFilters()
    query = db.query(
        FactSentiment.aspect_category,
        FactSentiment.sentiment_polarity,
        func.count().label("count"),
        func.avg(FactSentiment.fuzzy_crisp_score).label("avg_crisp_score"),
    )
    query = _apply_fact_filters(query, f)
    rows = (
        query.group_by(FactSentiment.aspect_category, FactSentiment.sentiment_polarity)
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


def top_aspects_by_polarity(
    db: Session, polarity: str, limit: int = 10, filters: AnalyticsFilters = None
) -> list[dict]:
    """Aspect leaderboard for a single polarity - shared by top_negative_aspects/top_positive_aspects."""
    f = filters or AnalyticsFilters()
    query = db.query(
        FactSentiment.aspect_term,
        func.count().label("count"),
        func.avg(FactSentiment.fuzzy_crisp_score).label("avg_crisp_score"),
    ).filter(FactSentiment.sentiment_polarity == polarity)
    query = _apply_fact_filters(query, f)
    rows = (
        query.group_by(FactSentiment.aspect_term)
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


def top_negative_aspects(db: Session, limit: int = 10, filters: AnalyticsFilters = None) -> list[dict]:
    return top_aspects_by_polarity(db, "negative", limit, filters)


def top_positive_aspects(db: Session, limit: int = 10, filters: AnalyticsFilters = None) -> list[dict]:
    return top_aspects_by_polarity(db, "positive", limit, filters)


def sentiment_over_time(db: Session, filters: AnalyticsFilters = None) -> list[dict]:
    f = filters or AnalyticsFilters()
    query = db.query(
        FactSentiment.date_id,
        FactSentiment.sentiment_polarity,
        func.count().label("count"),
        func.avg(FactSentiment.fuzzy_crisp_score).label("avg_crisp_score"),
    )
    query = _apply_fact_filters(query, f)
    rows = (
        query.group_by(FactSentiment.date_id, FactSentiment.sentiment_polarity)
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


def reviews_over_time(db: Session, filters: AnalyticsFilters = None) -> list[dict]:
    """Distinct review count per day - used for the Visão Geral sparkline (sentiment_over_time counts aspect mentions, not reviews)."""
    f = filters or AnalyticsFilters()
    query = db.query(
        FactSentiment.date_id,
        func.count(func.distinct(FactSentiment.id_review)).label("count"),
    )
    query = _apply_fact_filters(query, f)
    rows = query.group_by(FactSentiment.date_id).order_by(FactSentiment.date_id).all()
    return [{"date": str(r.date_id), "count": r.count} for r in rows]


def seasonality(db: Session, filters: AnalyticsFilters = None) -> list[dict]:
    """
    Aspect-mention volume by day-of-week x month, using DimCalendar's date
    hierarchy (previously unused beyond the plain date range filter).
    day_of_week follows Postgres EXTRACT(DOW): 0=Sunday..6=Saturday.
    """
    f = filters or AnalyticsFilters()
    # Group by the extract() expressions themselves (not their string labels):
    # DimCalendar already has a real "month" column, so grouping by the
    # string label "month" resolves to that column instead of our alias and
    # Postgres rejects the query.
    dow_expr = func.extract("dow", DimCalendar.date)
    month_expr = func.extract("month", DimCalendar.date)
    query = db.query(
        dow_expr.label("day_of_week"),
        month_expr.label("month"),
        func.count(FactSentiment.fact_id).label("count"),
    ).join(DimCalendar, DimCalendar.date_id == FactSentiment.date_id)
    query = _apply_fact_filters(query, f)
    rows = query.group_by(dow_expr, month_expr).all()
    return [
        {"day_of_week": int(r.day_of_week), "month": int(r.month), "count": r.count}
        for r in rows
    ]


def overview_kpis(db: Session, filters: AnalyticsFilters = None) -> dict:
    f = filters or AnalyticsFilters()

    def _filtered(query):
        return _apply_fact_filters(query, f)

    total_reviews = _filtered(db.query(func.count(func.distinct(FactSentiment.id_review)))).scalar() or 0
    total_aspects = _filtered(db.query(func.count(FactSentiment.fact_id))).scalar() or 0
    avg_score = _filtered(db.query(func.avg(FactSentiment.fuzzy_crisp_score))).scalar()

    polarity_counts = dict(
        _filtered(db.query(FactSentiment.sentiment_polarity, func.count(FactSentiment.fact_id)))
        .group_by(FactSentiment.sentiment_polarity)
        .all()
    )

    def pct(label: str) -> float:
        return round(100 * polarity_counts.get(label, 0) / total_aspects, 1) if total_aspects else 0.0

    return {
        "total_reviews": total_reviews,
        "total_aspects": total_aspects,
        "avg_crisp_score": float(avg_score) if avg_score is not None else None,
        "pct_positive": pct("positive"),
        "pct_neutral": pct("neutral"),
        "pct_negative": pct("negative"),
    }


def restaurant_performance(db: Session, filters: AnalyticsFilters = None) -> list[dict]:
    f = filters or AnalyticsFilters()
    query = db.query(
        DimRestaurant.id_restaurant,
        DimRestaurant.name,
        DimRestaurant.district,
        DimRestaurant.inspection_grade,
        func.avg(FactSentiment.fuzzy_crisp_score).label("avg_crisp_score"),
        func.count(func.distinct(FactSentiment.id_review)).label("review_count"),
    ).join(FactSentiment, FactSentiment.id_restaurant == DimRestaurant.id_restaurant)
    query = _apply_fact_filters(query, f, joined_restaurant=True)
    rows = (
        query.group_by(
            DimRestaurant.id_restaurant, DimRestaurant.name, DimRestaurant.district, DimRestaurant.inspection_grade
        )
        .order_by(desc("avg_crisp_score"))
        .all()
    )
    return [
        {
            "id_restaurant": r.id_restaurant,
            "name": r.name,
            "district": r.district,
            "inspection_grade": r.inspection_grade,
            "avg_crisp_score": float(r.avg_crisp_score) if r.avg_crisp_score is not None else None,
            "review_count": r.review_count,
        }
        for r in rows
    ]


def district_performance(db: Session, filters: AnalyticsFilters = None) -> list[dict]:
    """
    Per-district aggregate for the Visão Geral map: review count, average
    fuzzy score and sentiment-polarity breakdown. Callers that want the map
    to ignore the district filter (it always shows every district) should
    build `filters` with `districts=[]` regardless of what's selected in the
    sidebar - every other field is still honoured normally.
    """
    f = filters or AnalyticsFilters()

    def _filtered(query):
        return _apply_fact_filters(query, f, joined_restaurant=True).filter(DimRestaurant.district.isnot(None))

    totals = (
        _filtered(
            db.query(
                DimRestaurant.district,
                func.count(func.distinct(FactSentiment.id_review)).label("review_count"),
                func.avg(FactSentiment.fuzzy_crisp_score).label("avg_crisp_score"),
            ).join(FactSentiment, FactSentiment.id_restaurant == DimRestaurant.id_restaurant)
        )
        .group_by(DimRestaurant.district)
        .all()
    )

    polarity_rows = (
        _filtered(
            db.query(
                DimRestaurant.district,
                FactSentiment.sentiment_polarity,
                func.count().label("count"),
            ).join(FactSentiment, FactSentiment.id_restaurant == DimRestaurant.id_restaurant)
        )
        .group_by(DimRestaurant.district, FactSentiment.sentiment_polarity)
        .all()
    )
    polarity_by_district: dict[str, dict[str, int]] = {}
    for r in polarity_rows:
        polarity_by_district.setdefault(r.district, {})[r.sentiment_polarity] = r.count

    results = []
    for r in totals:
        pol = polarity_by_district.get(r.district, {})
        total_aspects = sum(pol.values())

        def pct(label: str) -> float:
            return round(100 * pol.get(label, 0) / total_aspects, 1) if total_aspects else 0.0

        results.append(
            {
                "district": r.district,
                "review_count": r.review_count,
                "avg_crisp_score": float(r.avg_crisp_score) if r.avg_crisp_score is not None else None,
                "pct_positive": pct("positive"),
                "pct_negative": pct("negative"),
                "pct_neutral": pct("neutral"),
            }
        )
    return results


# ── Recent activity (Submeter Review page) ────────────────────────

def recent_reviews(db: Session, limit: int = 10) -> list[dict]:
    """Most recently submitted reviews, one row per review with its aggregated sentiment."""
    rows = (
        db.query(
            DimReview.id_review,
            DimReview.text,
            DimReview.created_at,
            DimRestaurant.name.label("restaurant_name"),
            func.avg(FactSentiment.fuzzy_crisp_score).label("avg_crisp_score"),
            func.count(FactSentiment.fact_id).label("aspect_count"),
        )
        .join(FactSentiment, FactSentiment.id_review == DimReview.id_review)
        .join(DimRestaurant, DimRestaurant.id_restaurant == FactSentiment.id_restaurant)
        .group_by(DimReview.id_review, DimReview.text, DimReview.created_at, DimRestaurant.name)
        .order_by(desc(DimReview.created_at))
        .limit(limit)
        .all()
    )
    return [
        {
            "id_review": r.id_review,
            "text": r.text,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "restaurant_name": r.restaurant_name,
            "avg_crisp_score": float(r.avg_crisp_score) if r.avg_crisp_score is not None else None,
            "aspect_count": r.aspect_count,
        }
        for r in rows
    ]


def reviews_submitted_today(db: Session) -> int:
    today = date.today()
    return (
        db.query(func.count(func.distinct(DimReview.id_review)))
        .filter(func.date(DimReview.created_at) == today)
        .scalar()
        or 0
    )
