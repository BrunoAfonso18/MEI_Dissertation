import io
import os
import pandas as pd
from contextlib import asynccontextmanager
from datetime import date, datetime

from fastapi import FastAPI, HTTPException, Depends, Query, UploadFile, File
from sqlalchemy.orm import Session

from schemas import (
    QueryRequest,
    ReviewRequest,
    ReviewResponse,
    AspectResult,
    BulkUploadResponse,
    RestaurantResponse,
)
from database.connection import get_db
from database.repository import (
    AnalyticsFilters,
    add_fact_sentiment,
    load_or_create_dimensions,
    create_dim_review,
    district_performance,
    get_all_restaurants,
    recent_reviews,
    reviews_over_time,
    reviews_submitted_today,
    seasonality,
    sentiment_by_category,
    top_negative_aspects,
    top_positive_aspects,
    sentiment_over_time,
    overview_kpis,
    restaurant_performance,
)
from models.extractor import AspectExtractor
from models.category import CategoryClassifier
from models.fuzzy_sentiment import fuzzy_analyzer, polarity_to_scores

models = {}


MODEL_PATH = os.getenv("MODEL_PATH", "./absa_model_final")


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f" Loading models from {MODEL_PATH}...")
    models["extractor"] = AspectExtractor(MODEL_PATH)
    models["category"]  = CategoryClassifier()
    print(" Models ready")
    yield
    models.clear()


app = FastAPI(
    title="ABSA API",
    description="Aspect-Based Sentiment Analysis for Restaurant Reviews",
    version="2.0.0",
    lifespan=lifespan,
)


# ── Helpers ──────────────────────────────────────────────────────

def _analyse_text(text: str, restaurant_id: int, db: Session) -> tuple[int, list[dict]]:
    """
    Core pipeline: extract aspects → sentiment → fuzzy → category → save.
    Returns (review_id, list_of_aspect_dicts).
    """
    now = datetime.now()

    # 1. Create review dimension (auto-generated ID)
    dim_review = create_dim_review(db, text=text, source="api", language="pt", created_at=now)

    # 2. Ensure restaurant + calendar dimensions exist
    load_or_create_dimensions(db, restaurant_id=restaurant_id, created_at=now)

    # 3. Extract aspects and opinions
    extracted = models["extractor"].predict(text)
    aspects   = extracted["aspects"]
    opinions  = extracted["opinions"]

    polarities  = extracted["polarities"]
    confidences = extracted["confidences"]

    # One batched zero-shot call for every aspect in the review, instead of
    # one call per aspect - this is what dominated submission latency.
    categories = models["category"].predict_batch(aspects, text)

    results = []
    for i, aspect in enumerate(aspects):
        opinion    = opinions[i] if i < len(opinions) else None
        polarity   = polarities[i]  if i < len(polarities)  else "neutral"
        confidence = confidences[i] if i < len(confidences) else 0.5

        sentiment_raw = polarity_to_scores(polarity, confidence)
        fuzzy_result  = fuzzy_analyzer.analyze(sentiment_raw)
        category      = categories[i]

        aspect_data = {
            "aspect_term":       aspect,
            "opinion_term":      opinion,
            "pos_score":         fuzzy_result["positive_score"],
            "neut_score":        fuzzy_result["neutral_score"],
            "neg_score":         fuzzy_result["negative_score"],
            "fuzzy_crisp_score": fuzzy_result["defuzzified_score"],
            "sentiment_polarity": fuzzy_result["sentiment_label"],
            "confidence":        fuzzy_result["defuzzified_score"],
            "aspect_category":   category,
        }
        results.append(aspect_data)

        add_fact_sentiment(
            db,
            review_id=dim_review.id_review,
            restaurant_id=restaurant_id,
            created_at=now,
            aspect_data=aspect_data,
        )

    db.commit()
    return dim_review.id_review, results


# ── Health ────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "models_loaded": list(models.keys())}


# ── Restaurants ───────────────────────────────────────────────────

@app.get("/restaurants", response_model=list[RestaurantResponse])
async def list_restaurants(db: Session = Depends(get_db)):
    """List all registered restaurants."""
    return get_all_restaurants(db)


# ── Single review query ───────────────────────────────────────────

@app.post("/query", response_model=ReviewResponse)
async def query(request: QueryRequest, db: Session = Depends(get_db)):
    """
    Analyze a single restaurant review.
    Only the review text and restaurant ID are required.
    """
    try:
        review_id, results = _analyse_text(request.text, request.restaurant_id, db)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return ReviewResponse(
        id=review_id,
        restaurant_id=request.restaurant_id,
        text=request.text,
        source="api",
        language="pt",
        created_at=datetime.now(),
        aspects=[AspectResult(**r) for r in results],
    )


# ── Recent activity (Submeter Review page) ─────────────────────────

@app.get("/reviews/recent")
async def get_recent_reviews(limit: int = 10, db: Session = Depends(get_db)):
    """Feed of the most recently submitted reviews, plus how many were submitted today."""
    return {
        "reviews": recent_reviews(db, limit=limit),
        "today_count": reviews_submitted_today(db),
    }


# ── Bulk CSV upload ────────────────────────────────────────────────

@app.post("/reviews/upload", response_model=BulkUploadResponse)
async def upload_reviews(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    Upload a CSV file with columns: restaurant_id, text.
    All reviews are processed through the ABSA pipeline and saved to the DW.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="O ficheiro deve ser um CSV.")

    content = await file.read()
    try:
        df = pd.read_csv(io.StringIO(content.decode("utf-8")))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao ler o CSV: {e}")

    required = {"restaurant_id", "text"}
    if not required.issubset(df.columns):
        raise HTTPException(
            status_code=400,
            detail=f"O CSV deve ter as colunas: {required}. Encontradas: {set(df.columns)}",
        )

    total     = len(df)
    processed = 0
    failed    = 0
    errors    = []

    for idx, row in df.iterrows():
        try:
            restaurant_id = int(row["restaurant_id"])
            text          = str(row["text"]).strip()
            if not text:
                raise ValueError("Texto vazio")

            _analyse_text(text, restaurant_id, db)
            processed += 1
        except Exception as e:
            failed += 1
            errors.append(f"Linha {idx + 2}: {e}")

    return BulkUploadResponse(
        total=total,
        processed=processed,
        failed=failed,
        errors=errors[:20],  # cap error list to avoid huge responses
    )


# ── Analytics endpoints ───────────────────────────────────────────
#
# Every endpoint below shares the same filter set (get_analytics_filters):
# restaurant, district, category (cuisine), inspection grade, sentiment
# polarity, aspect category and a date range - matching the dashboard's
# filter sidebar. Each field accepts multiple values (OR/IN). Charts fire a
# request with whatever's selected, and the query is built dynamically
# against the star schema (FactSentiment joined with DimRestaurant when a
# restaurant-level filter is set) accordingly.

def get_analytics_filters(
    start_date: date | None = None,
    end_date: date | None = None,
    restaurant_id: list[int] | None = Query(None),
    district: list[str] | None = Query(None),
    category: list[str] | None = Query(None),
    inspection_grade: list[str] | None = Query(None),
    sentiment_polarity: list[str] | None = Query(None),
    aspect_category: list[str] | None = Query(None),
    aspect_term: list[str] | None = Query(None),
) -> AnalyticsFilters:
    return AnalyticsFilters(
        start_date=start_date,
        end_date=end_date,
        restaurant_ids=restaurant_id or [],
        districts=district or [],
        categories=category or [],
        grades=inspection_grade or [],
        polarities=sentiment_polarity or [],
        aspect_categories=aspect_category or [],
        aspect_terms=aspect_term or [],
    )


@app.get("/analytics/sentiment-by-category")
async def get_sentiment_by_category(
    filters: AnalyticsFilters = Depends(get_analytics_filters),
    db: Session = Depends(get_db),
):
    return sentiment_by_category(db, filters)


@app.get("/analytics/top-negative-aspects")
async def get_top_negative(
    limit: int = 10,
    filters: AnalyticsFilters = Depends(get_analytics_filters),
    db: Session = Depends(get_db),
):
    return top_negative_aspects(db, limit=limit, filters=filters)


@app.get("/analytics/top-positive-aspects")
async def get_top_positive(
    limit: int = 10,
    filters: AnalyticsFilters = Depends(get_analytics_filters),
    db: Session = Depends(get_db),
):
    return top_positive_aspects(db, limit=limit, filters=filters)


@app.get("/analytics/seasonality")
async def get_seasonality(
    filters: AnalyticsFilters = Depends(get_analytics_filters),
    db: Session = Depends(get_db),
):
    return seasonality(db, filters)


@app.get("/analytics/sentiment-over-time")
async def get_sentiment_over_time(
    filters: AnalyticsFilters = Depends(get_analytics_filters),
    db: Session = Depends(get_db),
):
    return sentiment_over_time(db, filters)


@app.get("/analytics/reviews-over-time")
async def get_reviews_over_time(
    filters: AnalyticsFilters = Depends(get_analytics_filters),
    db: Session = Depends(get_db),
):
    return reviews_over_time(db, filters)


@app.get("/analytics/overview")
async def get_overview(
    filters: AnalyticsFilters = Depends(get_analytics_filters),
    db: Session = Depends(get_db),
):
    return overview_kpis(db, filters)


@app.get("/analytics/restaurant-performance")
async def get_restaurant_performance(
    filters: AnalyticsFilters = Depends(get_analytics_filters),
    db: Session = Depends(get_db),
):
    return restaurant_performance(db, filters)


@app.get("/analytics/district-performance")
async def get_district_performance(
    filters: AnalyticsFilters = Depends(get_analytics_filters),
    db: Session = Depends(get_db),
):
    return district_performance(db, filters)
