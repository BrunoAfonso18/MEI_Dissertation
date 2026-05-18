from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.orm import Session
from datetime import datetime
from schemas import ReviewRequest, ReviewResponse, AspectResult, FactSentimentResponse
from database.connection import get_db
from database.repository import (
    save_fact_sentiment,
    load_or_create_dimensions,
    sentiment_by_category,
    top_negative_aspects,
    sentiment_over_time,
)
from models.extractor import AspectExtractor
from models.sentiment import SentimentClassifier
from models.category import CategoryClassifier
from models.fuzzy_sentiment import fuzzy_analyzer

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
    description="Aspect-Based Sentiment Analysis for Restaurant Reviews",
    version="2.0.0",
    lifespan=lifespan,
)

@app.get("/health")
async def health():
    return {"status": "ok", "models_loaded": list(models.keys())}

@app.post("/analyze", response_model=ReviewResponse)
async def analyze(request: ReviewRequest, db: Session = Depends(get_db)):
    """
    Analyze a restaurant review:
    1. Extract aspects and opinions
    2. Classify sentiment (positive, neutral, negative)
    3. Apply fuzzy logic to compute crisp scores
    4. Save to DW dimensions and fact table
    """
    # Step 1: Extract aspects from text
    extracted = models["extractor"].predict(request.text)
    aspects   = extracted["aspects"]
    opinions  = extracted["opinions"]
    
    # Step 2: Load or create restaurant and date dimensions
    dim_restaurant = load_or_create_dimensions(
        db,
        restaurant_id=request.restaurant_id,
        created_at=datetime.now()
    )
    
    results = []
    for i, aspect in enumerate(aspects):
        opinion  = opinions[i] if i < len(opinions) else None
        
        # Step 3: Sentiment classification (raw scores)
        sentiment_raw = models["sentiment"].predict(aspect, request.text)
        
        # Step 4: Apply fuzzy logic to get crisp score
        fuzzy_result = fuzzy_analyzer.analyze(sentiment_raw)
        
        # Step 5: Category classification
        category = models["category"].predict(aspect, request.text)
        
        # Build aspect result with all metrics
        aspect_data = {
            "aspect_term":       aspect,
            "opinion_term":      opinion,
            "pos_score":         fuzzy_result["positive_score"],
            "neut_score":        fuzzy_result["neutral_score"],
            "neg_score":         fuzzy_result["negative_score"],
            "fuzzy_crisp_score": fuzzy_result["defuzzified_score"],  # Centroid method
            "sentiment_polarity": fuzzy_result["sentiment_label"],
            "confidence":        fuzzy_result["defuzzified_score"],
            "aspect_category":   category,
        }
        results.append(aspect_data)
        
        # Step 6: Save to fact_sentiment table
        fact_sentiment = save_fact_sentiment(
            db,
            review_id=request.review_id,
            restaurant_id=request.restaurant_id,
            created_at=datetime.now(),
            aspect_data=aspect_data
        )
    
    return ReviewResponse(
        id=request.review_id,
        restaurant_id=request.restaurant_id,
        text=request.text,
        source=request.source,
        language=request.language,
        created_at=datetime.now(),
        aspects=[AspectResult(**r) for r in results]
    )

# ── Analytics endpoints ─────────────────────────────────────────
@app.get("/analytics/sentiment-by-category")
async def get_sentiment_by_category(db: Session = Depends(get_db)):
    """Get sentiment distribution by aspect category"""
    return sentiment_by_category(db)

@app.get("/analytics/top-negative-aspects")
async def get_top_negative(limit: int = 10, db: Session = Depends(get_db)):
    """Get most complained about aspects"""
    return top_negative_aspects(db, limit=limit)

@app.get("/analytics/sentiment-over-time")
async def get_sentiment_over_time(db: Session = Depends(get_db)):
    """Get sentiment trends over time"""
    return sentiment_over_time(db)