from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.orm import Session
from schemas import ReviewRequest, ReviewResponse, AspectResult
from database.connection import get_db
from database.repository import (
    save_review,
    save_aspects,
    sentiment_by_category,
    top_negative_aspects,
    sentiment_over_time,
)
from models.extractor import AspectExtractor
from models.sentiment import SentimentClassifier
from models.category import CategoryClassifier
from models.fuzzy_sentiment import fuzzy_analyzer

# ── Load models once at startup ─────────────────────────────────
models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(" Loading models...")
    models["extractor"] = AspectExtractor("./absa_model_final")
    models["sentiment"] = SentimentClassifier()
    models["category"]  = CategoryClassifier()
    print(" Models ready")
    yield
    models.clear()

app = FastAPI(
    title="ABSA API",
    description="Aspect-Based Sentiment Analysis",
    version="1.0.0",
    lifespan=lifespan,
)

@app.get("/health")
async def health():
    return {"status": "ok", "models_loaded": list(models.keys())}

# ── Analyze endpoint ────────────────────────────────────────────
@app.post("/analyze", response_model=ReviewResponse)
async def analyze(request: ReviewRequest, db: Session = Depends(get_db)):
    extracted = models["extractor"].predict(request.text)
    aspects   = extracted["aspects"]
    opinions  = extracted["opinions"]
    polarities = extracted.get("polarities", [])

    results = []
    for i, aspect in enumerate(aspects):
        opinion  = opinions[i] if i < len(opinions) else None
        sentiment_raw = models["sentiment"].predict(aspect, request.text)
        
        fuzzy_result = fuzzy_analyzer.analyze(sentiment_raw)
        
        category = models["category"].predict(aspect, request.text)
        results.append({
            "aspect_term":        aspect,
            "opinion_term":       opinion,
            "positive_score":   fuzzy_result["positive_score"],
            "neutral_score":    fuzzy_result["neutral_score"],
            "negative_score":   fuzzy_result["negative_score"],
            "sentiment_polarity": fuzzy_result["sentiment_label"],
            "confidence":         fuzzy_result["defuzzified_score"],
            "aspect_category":    category,
        })

    # 2. Persist to database
    review = save_review(db, text=request.text, source="api")
    save_aspects(db, review_id=review.id, aspects=results)

    return ReviewResponse(
        sentence=request.text,
        aspects=[AspectResult(**r) for r in results]
    )

# ── Analytics endpoints ─────────────────────────────────────────
@app.get("/analytics/sentiment-by-category")
async def get_sentiment_by_category(db: Session = Depends(get_db)):
    return sentiment_by_category(db)

@app.get("/analytics/top-negative-aspects")
async def get_top_negative(limit: int = 10, db: Session = Depends(get_db)):
    return top_negative_aspects(db, limit=limit)

@app.get("/analytics/sentiment-over-time")
async def get_sentiment_over_time(db: Session = Depends(get_db)):
    return sentiment_over_time(db)