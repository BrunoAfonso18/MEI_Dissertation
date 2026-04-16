from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.orm import Session
from database.connection import get_db
from database.repository import save_review, save_aspects, sentiment_by_category, top_negative_aspects

@app.post("/analyze", response_model=ReviewResponse)
async def analyze(request: ReviewRequest, db: Session = Depends(get_db)):
    # 1. Extract aspects
    extracted = models["extractor"].predict(request.text)
    aspects   = extracted["aspects"]
    opinions  = extracted["opinions"]

    results = []
    for i, aspect in enumerate(aspects):
        opinion  = opinions[i] if i < len(opinions) else None
        polarity = models["sentiment"].predict(aspect, request.text)
        category = models["category"].predict(aspect, request.text)
        results.append({
            "aspect_term":        aspect,
            "opinion_term":       opinion,
            "sentiment_polarity": polarity,
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