from pydantic import BaseModel
from typing import Optional
from datetime import datetime, date

class AspectResult(BaseModel):
    """Aspect sentiment analysis result"""
    aspect_term: Optional[str]
    opinion_term: Optional[str]
    pos_score: Optional[float]      # Fuzzy positive membership
    neut_score: Optional[float]     # Fuzzy neutral membership
    neg_score: Optional[float]      # Fuzzy negative membership
    fuzzy_crisp_score: Optional[float]  # Crisp score from centroid method
    sentiment_polarity: Optional[str]
    confidence: Optional[float]
    aspect_category: Optional[str]

    class Config:
        from_attributes = True


class ReviewRequest(BaseModel):
    """Request to analyze a restaurant review"""
    restaurant_id: int              # Restaurant being reviewed
    review_id: int                  # Source system review ID
    text: str                       # Review text to analyze
    source: str = "api"             # Data source
    language: str = "en"            # Language code


class ReviewResponse(BaseModel):
    """Response with analysis results"""
    id: int                         # Review ID
    restaurant_id: int              # Restaurant ID
    text: str                       # Original review text
    source: str                     # Data source
    language: str                   # Language
    created_at: datetime            # Analysis timestamp
    aspects: list[AspectResult]     # Detected aspects and sentiments

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