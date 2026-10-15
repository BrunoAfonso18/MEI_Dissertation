import asyncio
import io
import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from database.connection import get_db, SessionLocal
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
from models.fuzzy_inference import IMPLICIT_ASPECT, IMPLICIT_CATEGORY, score_aspect, score_implicit
from models.linguistic_adjustments import detect_language

models = {}


MODEL_PATH = os.getenv("MODEL_PATH", "./absa_model_final")

# Number of CSV rows processed at the same time by POST /reviews/upload.
# Kept within SQLAlchemy's default connection pool (5 + 10 overflow), since
# each worker holds one connection while processing its row.
BULK_MAX_WORKERS = int(os.getenv("BULK_MAX_WORKERS", min(4, os.cpu_count() or 1)))


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

    # 1. Ensure restaurant + calendar dimensions exist. Done before creating
    # the review, since create_dim_review commits immediately: an unknown
    # restaurant_id would otherwise leave an orphan review (no facts) behind
    # even though the row is reported as failed.
    load_or_create_dimensions(db, restaurant_id=restaurant_id, created_at=now)

    # 2. Create review dimension (auto-generated ID)
    dim_review = create_dim_review(db, text=text, source="api", language=detect_language(text), created_at=now)

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

        # Correções de negação/ironia (models/linguistic_adjustments.py) e
        # sistema difuso de Mamdani que mede a intensidade da opinião a partir
        # do léxico, do modificador de grau e da evidência do modelo
        # (models/fuzzy_inference.py).
        scored   = score_aspect(text, aspect, opinion, polarity, confidence)
        category = categories[i]

        aspect_data = {
            "aspect_term":       aspect,
            "opinion_term":      opinion,
            "pos_score":         scored["positive_score"],
            "neut_score":        scored["neutral_score"],
            "neg_score":         scored["negative_score"],
            "fuzzy_crisp_score": scored["crisp"],
            "sentiment_polarity": scored["label"],
            "confidence":        scored["certainty"],
            "aspect_category":   category,
            "negation_detected": scored["negation_detected"],
            "irony_detected":    scored["irony_detected"],
        }
        results.append(aspect_data)

        add_fact_sentiment(
            db,
            review_id=dim_review.id_review,
            restaurant_id=restaurant_id,
            created_at=now,
            aspect_data=aspect_data,
        )

    # Sem nenhum aspeto explícito ("Muito bom!"), a opinião global é inferida
    # só pela camada difusa e guardada no aspeto implícito "geral" (GENERAL).
    if not aspects:
        scored = score_implicit(text)
        if scored is not None:
            aspect_data = {
                "aspect_term":       IMPLICIT_ASPECT,
                "opinion_term":      scored["opinion_phrase"],
                "pos_score":         scored["positive_score"],
                "neut_score":        scored["neutral_score"],
                "neg_score":         scored["negative_score"],
                "fuzzy_crisp_score": scored["crisp"],
                "sentiment_polarity": scored["label"],
                "confidence":        scored["certainty"],
                "aspect_category":   IMPLICIT_CATEGORY,
                "negation_detected": scored["negation_detected"],
                "irony_detected":    scored["irony_detected"],
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
        language=detect_language(request.text),
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
async def upload_reviews(file: UploadFile = File(...)):
    """
    Upload a CSV file with columns: restaurant_id, text.
    All reviews are processed through the ABSA pipeline and saved to the DW,
    BULK_MAX_WORKERS rows at a time (see _run_bulk).
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

    # (CSV line number, restaurant_id, text) - line + 2 accounts for the
    # header row and 1-based numbering, as shown to the user.
    rows = [
        (idx + 2, row["restaurant_id"], row["text"])
        for idx, row in df.iterrows()
    ]

    # Run off the event loop so the API keeps answering other requests
    # (dashboard, single reviews) while a large CSV is being processed.
    processed, errors = await asyncio.to_thread(_run_bulk, rows)

    return BulkUploadResponse(
        total=len(rows),
        processed=processed,
        failed=len(errors),
        errors=[f"Linha {line}: {msg}" for line, msg in errors[:20]],  # cap error list to avoid huge responses
    )


def _process_bulk_row(restaurant_id, text) -> None:
    """
    Process one CSV row in a worker thread. A SQLAlchemy Session is not
    thread-safe, so each row gets its own (connections come from the
    engine's pool) instead of sharing the request's session.
    """
    if pd.isna(text) or not str(text).strip():
        raise ValueError("Texto vazio")
    restaurant_id = int(restaurant_id)
    text = str(text).strip()

    db = SessionLocal()
    try:
        _analyse_text(text, restaurant_id, db)
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def _run_bulk(rows: list[tuple]) -> tuple[int, list[tuple[int, str]]]:
    """
    Process the CSV rows concurrently with a thread pool. Threads (rather
    than processes) are enough here: the time per review is spent in torch
    (which releases the GIL during the forward pass) and in round trips to
    PostgreSQL (I/O), and threads share the already-loaded models instead
    of loading a copy per process.
    Returns (processed_count, [(line, error_message), ...] sorted by line).
    """
    processed = 0
    errors    = []
    with ThreadPoolExecutor(max_workers=BULK_MAX_WORKERS) as pool:
        futures = {
            pool.submit(_process_bulk_row, restaurant_id, text): line
            for line, restaurant_id, text in rows
        }
        for future in as_completed(futures):
            try:
                future.result()
                processed += 1
            except Exception as e:
                errors.append((futures[future], str(e)))

    errors.sort()
    return processed, errors


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
