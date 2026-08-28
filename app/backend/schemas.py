from pydantic import BaseModel
from typing import Optional
from datetime import datetime, date


class AspectResult(BaseModel):
    aspect_term: Optional[str]
    opinion_term: Optional[str]
    pos_score: Optional[float]
    neut_score: Optional[float]
    neg_score: Optional[float]
    fuzzy_crisp_score: Optional[float]
    sentiment_polarity: Optional[str]
    confidence: Optional[float]
    aspect_category: Optional[str]
    negation_detected: Optional[bool] = False
    irony_detected: Optional[bool] = False

    class Config:
        from_attributes = True


class QueryRequest(BaseModel):
    """Simplified query: only the review text and restaurant ID are needed."""
    text: str
    restaurant_id: int


class ReviewRequest(BaseModel):
    """Full review request (kept for backwards compatibility)."""
    restaurant_id: int
    review_id: int
    text: str
    source: str = "api"
    language: str = "pt"


class ReviewResponse(BaseModel):
    id: int
    restaurant_id: int
    text: str
    source: str
    language: str
    created_at: datetime
    aspects: list[AspectResult]

    class Config:
        from_attributes = True


class BulkUploadResponse(BaseModel):
    """Summary response for CSV bulk upload."""
    total: int
    processed: int
    failed: int
    errors: list[str] = []


class RestaurantResponse(BaseModel):
    id_restaurant: int
    name: Optional[str]
    district: Optional[str]
    category: Optional[str]
    address: Optional[str]
    inspection_grade: Optional[str]

    class Config:
        from_attributes = True


class DimCalendarResponse(BaseModel):
    date_id: date
    date: date
    week: date
    month: date
    year: int

    class Config:
        from_attributes = True


class DimReviewResponse(BaseModel):
    id_review: int
    source: Optional[str]
    text: Optional[str]
    language: Optional[str]
    created_at: Optional[datetime]

    class Config:
        from_attributes = True


class DimRestaurantResponse(BaseModel):
    id_restaurant: int
    district: Optional[str]
    category: Optional[str]
    address: Optional[str]
    name: Optional[str]
    inspection_grade: Optional[str]

    class Config:
        from_attributes = True


class FactSentimentResponse(BaseModel):
    fact_id: int
    id_review: int
    id_restaurant: int
    date_id: date
    aspect_term: Optional[str]
    opinion_term: Optional[str]
    aspect_category: Optional[str]
    fuzzy_crisp_score: Optional[float]
    sentiment_polarity: Optional[str]
    confidence_score: Optional[float]
    created_at: Optional[datetime]

    class Config:
        from_attributes = True


class FactSentimentCreate(BaseModel):
    id_review: int
    id_restaurant: int
    date_id: date
    aspect_term: Optional[str] = None
    opinion_term: Optional[str] = None
    aspect_category: Optional[str] = None
    fuzzy_crisp_score: Optional[float] = None
    sentiment_polarity: Optional[str] = None
    confidence_score: Optional[float] = None
    created_at: Optional[datetime] = None
